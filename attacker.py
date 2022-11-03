import torch 
import torch.nn as nn 
import torch.nn.functional as F
IMAGE_SCALE = 2.0/255


def get_kernel(size, nsig, mode='gaussian', device='cuda:0'):
    if mode == 'gaussian':
        # since we have to normlize all the numbers 
        # there is no need to calculate the const number like \pi and \sigma.
        vec = torch.linspace(-nsig, nsig, steps=size).to(device)
        vec = torch.exp(-vec*vec/2)
        res = vec.view(-1, 1) @ vec.view(1, -1) 
        res = res / torch.sum(res)
    elif mode == 'linear':
        # originally, res[i][j] = (1-|i|/(k+1)) * (1-|j|/(k+1))
        # since we have to normalize it
        # calculate res[i][j] = (k+1-|i|)*(k+1-|j|)
        vec = (size+1)/2 - torch.abs(torch.arange(-(size+1)/2, (size+1)/2+1, step=1)).to(device)
        res = vec.view(-1, 1) @ vec.view(1, -1) 
        res = res / torch.sum(res)
    else:
        raise ValueError("no such mode in get_kernel.")
    return res


def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):

        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        # dealing with out-of-bounded values
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
  
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


class NoOpAttacker():
    
    def attack(self, image, label, model):
        return image, -torch.ones_like(label)


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, kernel_size=15, prob_start_from_clean=0.0, translation=False, device='cuda:0'):
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size*IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean
        self.device=device
        self.translation = translation
        if translation:
            # this is equivalent to deepth wise convolution
            # details can be found in the docs of Conv2d.
            # "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also termed in literature as depthwise convolution."
            self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=(kernel_size-1)//2, bias=False, groups=3).to(self.device)
            self.gkernel = get_kernel(kernel_size, nsig=3, device=self.device).to(self.device)
            self.conv.weight = self.gkernel
    
    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=1000)
        return (label + label_offset) % 1000

    def attack(self, image_clean, label, model, xfm=None, ifm=None, criterion=nn.CrossEntropyLoss, original=False):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label)
        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)

        ori_images = image_clean.clone().detach()

        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)
        
        start_from_noise_index = (torch.randn([])>self.prob_start_from_clean).float() 
        start_adv = image_clean + start_from_noise_index * init_start

        adv = start_adv
        for i in range(self.num_iter):
            adv.requires_grad = True
            logits = model(adv)
            losses = criterion(logits, target_label)
            g = torch.autograd.grad(losses, adv, 
                                    retain_graph=False, create_graph=False)[0]
            if self.translation:
                g = self.conv(g)
            if original:
                adv = adv + torch.sign(g)*self.step_size
            else:
                adv = adv - torch.sign(g) * self.step_size
            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()

        # with torch.no_grad():
        #     logits = model(adv)
        #     output_loss = criterion(logits, target_label, reduce=False, reduction='none')
        
        return adv, target_label, losses



class FFTAttacker():
    def __init__(self, num_iter, epsilon, lr=0.005, prob_start_from_clean=0.0, device='cuda:0'):
        # step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        # self.epsilon = epsilon * IMAGE_SCALE
        # self.step_size = step_size*IMAGE_SCALE
        # self.prob_start_from_clean = prob_start_from_clean
        self.device = device
        # self.translation = translation
        self.lr = lr

        self.rfft = torch.fft.rfft2
        self.irfft = torch.fft.irfft2

        self.gs_sigma = 70
    
    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=1000)
        return (label + label_offset) % 1000

    def attack(self, image_clean, label, model, original=True):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label)
        f_map = self.rfft(image_clean,dim=(-2,-1))

        # normalize the fft map
        f_top_w = torch.arange(0,113).repeat((len(image_clean),1,113,1))
        f_top_h = f_top_w.permute((0,1,3,2))
        f_top_weight = torch.exp(-(torch.pow(f_top_w,2)+torch.pow(f_top_h,2))/(2*self.gs_sigma*self.gs_sigma))[:,:,:-1,:]
        f_down_weight = torch.flip(f_top_weight,(2,))
        f_weight_train = torch.cat((f_top_weight,f_down_weight),2).clone().detach().cuda().requires_grad_(True)
        f_weight_variable = f_weight_train.expand(-1,3,-1,-1)
        f_weight_const = 1 / f_weight_variable


        optimizer=torch.optim.SGD([f_weight_train], lr=self.lr)

        for i in range(self.num_iter):
            f_weight = f_weight_train * f_weight_const
            adv = self.irfft(f_map*f_weight,dim=(-2,-1))
            adv = torch.clamp(adv,min=-1,max=1)

            logits = model(adv)
            if original:
                loss = -F.cross_entropy(logits, target_label)
            else:
                loss = F.cross_entropy(logits, target_label)
            loss.backward(retain_graph=True)
            optimizer.step()
        
        adv = self.irfft(f_map*f_weight_train*f_weight_const,dim=(-2,-1))
        adv = torch.clamp(adv,min=-1,max=1)

        return adv, target_label


class WTAttacker():
    def __init__(self, num_iter, coeff_layers=6, device='cuda:0'):
        self.device = device
        self.coeff_layers = coeff_layers
        self.num_iter = num_iter
        # self.convertor = RGB_HSV()

    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=1000)
        return (label + label_offset) % 1000


    def attack(self, image_clean, label, model, xfm=None, ifm=None, criterion=nn.CrossEntropyLoss, original=False):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label)
        
        yL,yH = xfm(image_clean)
        yL_shape = list(yL.shape)
        yH_shape = [list(yHi.shape) for yHi in yH]

        # weights
        yL_weight_train=nn.Parameter(torch.ones(size=(yL_shape[0],3,yL_shape[2],yL_shape[3])).cuda())
        yH_weight_train=nn.ParameterList([nn.Parameter(torch.ones(size=(yH_shape[i][0],3,yH_shape[i][2],yH_shape[i][3],yH_shape[i][4])).cuda()) for i in range(self.coeff_layers)])

        # # wt_test8
        # yH_lr=[0.36,0.02,0.01,0.01,0.005,0.0005]
        # yL_lr=0.0005
        # # yH_lr=[0.005,0.002,0.0004,0.0003,0.0002,0.0001]
        # # yL_lr=0.0001

        # wt_test19
        # yH_lr=[0.5,0.07,0.05,0.03,0.02,0.01]
        # yL_lr=0.01

        # wt_test16(mid)
        # yH_lr=[0.05,0.07,0.09,0.07,0.05,0.03]
        # yL_lr=0.01

        # wt_test14
        yH_lr=[0.30,0.05,0.04,0.03,0.02,0.015]
        yL_lr=0.015

        if not (len(yH_lr)==self.coeff_layers):
            raise AssertionError ("Learning rate list must match the coeff_layers.")

        # optim_params=[
        #     {'params': yL_weight_train, 'lr': yL_lr},
        # ]
        # optim_params+=[{'params': yH_weight_train[i], 'lr': yH_lr[i]} for i in range(self.coeff_layers)]
        # optimizer=torch.optim.Adam(optim_params)

        iteration=0

        while iteration < self.num_iter:
            # optimizer.zero_grad()
            # yL_weight=torch.cat((yL_weight_h_train,yL_weight_sv_train),1)
            # yH_weight=[torch.cat((yH_weight_h_train[i],yH_weight_sv_train[i]),1) for i in range(self.coeff_layers)]
            reconstructed_tensor=ifm((yL*yL_weight_train, [yH[i]*yH_weight_train[i] for i in range(self.coeff_layers)]))
            reconstructed_tensor = torch.clamp(reconstructed_tensor,min=-1.0,max=1.0)
            logits = model(reconstructed_tensor)
            if original:
                loss = -criterion(logits, target_label)
            else:
                loss = criterion(logits, target_label)

            loss.backward()
            # optimizer.step()
            iteration += 1

        # adv = ifm((yL*yL_weight_train, [yH[i]*yH_weight_train[i] for i in range(self.coeff_layers)]))
        adv = ifm((yL*(yL_weight_train-yL_lr*yL_weight_train.grad.detach()), [yH[i]*(yH_weight_train[i]-yH_lr[i]*yH_weight_train[i].grad.detach()) for i in range(self.coeff_layers)]))
        adv = torch.clamp(adv,min=-1.0,max=1.0).detach()
        
        # with torch.no_grad():
        #     logits = model(adv)
        #     output_loss = criterion(logits, target_label, reduce=False, reduction='none')

        return adv, target_label, loss

class Gaussian_Attacker():
    def __init__(self, device='cuda:0'):
        self.device = device
    
    def attack(self, image_clean, mean=0.0, var=0.001):
        # noise = np.random.normal(mean, var ** 0.5, image_clean.shape)
        # noise_tensor = torch.from_numpy(noise).to(image_clean.device).float()
        noise = torch.zeros(image_clean.shape, dtype=torch.float32, device=image_clean.device)
        noise = noise + (var**0.5)*torch.randn(image_clean.shape, device=image_clean.device)
        adv = image_clean + noise
        adv = torch.clamp(adv,min=-1.0,max=1.0).detach()
        return adv

class EnsembleAttacker():
    def __init__(self, num_iter, epsilon, step_size, coeff_layers=6, kernel_size=15, prob_start_from_clean=0.0, translation=False, device='cuda:0'):
        self.coeff_layers = coeff_layers
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size*IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean
        self.device=device
        self.translation = translation
        if translation:
            # this is equivalent to deepth wise convolution
            # details can be found in the docs of Conv2d.
            # "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also termed in literature as depthwise convolution."
            self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=(kernel_size-1)//2, bias=False, groups=3).to(self.device)
            self.gkernel = get_kernel(kernel_size, nsig=3, device=self.device).to(self.device)
            self.conv.weight = self.gkernel
        self.convertor = RGB_HSV()

    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=1000)
        return (label + label_offset) % 1000

    def attack(self, image_clean, label, model, _mean, _std, xfm=None, ifm=None, original=False):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label)
        
        # PGD setting
        lower_bound = torch.clamp(image_clean - self.epsilon, min=0., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=0., max=1.)

        adv_start = image_clean.clone().detach()
        adv_start.require_grad = True
        # WT setting
        hsv_tensor = self.convertor.rgb_to_hsv(adv_start)
        # hsv_tensor = image_clean
        yL,yH = xfm(hsv_tensor)
        yL_shape = list(yL.shape)
        yH_shape = [list(yHi.shape) for yHi in yH]

        # h and sv weights
        yL_weight_h_train=nn.Parameter(torch.ones(size=(yL_shape[0],1,yL_shape[2],yL_shape[3])).cuda())
        yH_weight_h_train=nn.ParameterList([nn.Parameter(torch.ones(size=(yH_shape[i][0],1,yH_shape[i][2],yH_shape[i][3],yH_shape[i][4])).cuda()) for i in range(self.coeff_layers)])
        yL_weight_sv_train=nn.Parameter(torch.ones(size=(yL_shape[0],2,yL_shape[2],yL_shape[3])).cuda())
        yH_weight_sv_train=nn.ParameterList([nn.Parameter(torch.ones(size=(yH_shape[i][0],2,yH_shape[i][2],yH_shape[i][3],yH_shape[i][4])).cuda()) for i in range(self.coeff_layers)])

        # test2 target hsv adam CE
        # yH_lr_h=[0.005,0.002,0.001,0.0003,0.0002,0.0001]
        # yL_lr_h=0.0001
        # yH_lr_sv=[0.05,0.02,0.01,0.003,0.002,0.001]
        # yL_lr_sv=0.001
        # yH_lr_h=[i*2 for i in yH_lr_h]
        # yL_lr_h=yL_lr_h*2
        # yH_lr_sv=[i*3 for i in yH_lr_sv]
        # yL_lr_sv=yL_lr_sv*3

        # wt_test5_sv0_8
        yH_lr_h=[0.02,0.012,0.008,0.0002,0.0002,0.0002]
        yL_lr_h=0.0002
        yH_lr_sv=[0.168,0.08,0.04,0.001,0.001,0.001]
        yL_lr_sv=0.001
        yH_lr_h=[i*2 for i in yH_lr_h]
        yL_lr_h=yL_lr_h*2
        yH_lr_sv=[i*3 for i in yH_lr_sv]
        yL_lr_sv=yL_lr_sv*3

        if not (len(yH_lr_h)==self.coeff_layers and len(yH_lr_sv)==self.coeff_layers):
            raise AssertionError ("Learning rate list must match the coeff_layers.")

        optim_params=[
            {'params': yL_weight_h_train, 'lr': yL_lr_h},
            {'params': yL_weight_sv_train, 'lr': yL_lr_sv}
        ]
        optim_params+=[{'params': yH_weight_h_train[i], 'lr': yH_lr_h[i]} for i in range(self.coeff_layers)]
        optim_params+=[{'params': yH_weight_sv_train[i], 'lr': yH_lr_sv[i]} for i in range(self.coeff_layers)]
        optimizer=torch.optim.Adam(optim_params)

        iteration=0

        while iteration < self.num_iter:
            adv_start.requires_grad = True
            optimizer.zero_grad()
            # hsv
            yL_weight=torch.cat((yL_weight_h_train,yL_weight_sv_train),1)
            yH_weight=[torch.cat((yH_weight_h_train[i],yH_weight_sv_train[i]),1) for i in range(self.coeff_layers)]

            reconstructed_hsv_tensor=ifm((yL*yL_weight, [yH[i]*yH_weight[i] for i in range(self.coeff_layers)]))
            # reconstructed_tensor.requires_grad = True

            # print(reconstructed_hsv_tensor)
            reconstructed_hsv_tensor = torch.clamp(reconstructed_hsv_tensor,min=0.0,max=1.0)
            reconstructed_tensor = self.convertor.hsv_to_rgb(reconstructed_hsv_tensor)
            # reconstructed_tensor = reconstructed_hsv_tensor
            reconstructed_tensor = torch.clamp(reconstructed_tensor,min=0.0,max=1.0)

            logits = model((reconstructed_tensor-_mean)/_std)
            if original:
                loss = -F.cross_entropy(logits, target_label)
            else:
                loss = F.cross_entropy(logits, target_label)

            loss.backward()
            g = adv_start.grad

            # WT 
            optimizer.step()

            # PGD
            # g = reconstructed_tensor.grad
            if self.translation:
                g = self.conv(g)
            if original:
                pgd_adv = adv_start + torch.sign(g)*self.step_size
            else:
                pgd_adv = adv_start - torch.sign(g) * self.step_size
            pgd_adv = torch.where(pgd_adv > lower_bound, pgd_adv, lower_bound)
            pgd_adv = torch.where(pgd_adv < upper_bound, pgd_adv, upper_bound).detach()

            iteration += 1

        # PGD
        logits = model((pgd_adv-_mean)/_std)
        output_loss1 = F.cross_entropy(logits, target_label, reduce=False, reduction='none')

        # WT
        yL_weight_adv = torch.cat((yL_weight_h_train,yL_weight_sv_train), 1)
        yH_weight_adv = [torch.cat((yH_weight_h_train[i],yH_weight_sv_train[i]), 1) for i in range(self.coeff_layers)]
        hsv_adv = ifm((yL*yL_weight_adv, [yH[i]*yH_weight_adv[i] for i in range(self.coeff_layers)]))
        hsv_adv = torch.clamp(hsv_adv,min=0.0,max=1.0)
        rgb_adv = self.convertor.hsv_to_rgb(hsv_adv)
        # rgb_adv = hsv_adv
        rgb_adv = torch.clamp(rgb_adv,min=0.0,max=1.0)

        logits = model((rgb_adv-_mean)/_std)
        output_loss2 = F.cross_entropy(logits, target_label, reduce=False, reduction='none')

        # ensemble
        adv = torch.where(output_loss1.view(output_loss1.shape[0],1,1,1) <= output_loss2.view(output_loss2.shape[0],1,1,1), pgd_adv, rgb_adv)


        return adv, target_label, output_loss2

class VQAttacker():
    def __init__(self, num_iter, size, lr=1.0e-4):
        # self.device = device
        # self.coeff_layers = coeff_layers
        self.num_iter = num_iter
        self.size=size
        self.lr=lr
        # self.convertor = RGB_HSV()


    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=1000)
        return (label + label_offset) % 1000


    def attack(self, image_clean, label, vqmodel, src_model, original=False):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label)
        
        quant_t, quant_b, diff, _, _ = vqmodel.encode(image_clean)
        quant_t_adv = quant_t.detach().clone().requires_grad_(True)
        quant_b_adv = quant_b.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([quant_t_adv, quant_b_adv], lr=self.lr)

        # init_xadv = model.module.decode(quant_t_adv, quant_b_adv)
        # bg_noise = (img - init_xadv).clone().detach()
        iteration=0

        while iteration < self.num_iter:
            optimizer.zero_grad()
            dec = vqmodel.decode(quant_t_adv, quant_b_adv)
            dec = torch.clamp(dec,min=-1.0,max=1.0)

            # to regular norm
            mean_tensor=torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            std_tensor=torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            dec_norm=(dec*0.5+0.5-mean_tensor)/std_tensor
            
            input = F.interpolate(dec_norm, size=self.size, mode="bicubic")
            logits = src_model(input)
            if original:
                loss = -F.cross_entropy(logits, target_label)
            else:
                loss = F.cross_entropy(logits, target_label)

            loss.backward()
            optimizer.step()
            iteration += 1

        adv = vqmodel.decode(quant_t_adv, quant_b_adv).clamp_(-1.0, 1.0).detach()
        # adv = torch.clamp(adv,min=-1.0,max=1.0)
        # adv=F.interpolate(adv, size=self.size, mode="bicubic")
        
        # with torch.no_grad():
        #     logits = src_model(adv)
        #     output_loss = F.cross_entropy(logits, target_label, reduce=False, reduction='none')
        output_loss=0
        return adv, target_label, output_loss