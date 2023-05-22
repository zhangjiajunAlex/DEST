import torch
from .base_model import BaseModel
import torch.nn.functional as F

from . import network, base_function, external_function
from util import task
import itertools

class DEST(BaseModel):
    def name(self):
        return "DEST Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=5, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=20.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.visual_names = ['img_m', 'img_truth', 'img_out', 'merged_image', 'img_out2']
        self.value_names = ['u_m', 'sigma_m', 'u_prior', 'sigma_prior']
        self.model_names = ['ET', 'ES','G', 'D', 'cross_attenstion', 'fuse_s', 'fuse_t', 'G2']
        self.loss_names = ['kl_s', 'kl_t', 'app_G2', 'app_G1', 'img_dg', 'ad_l', 'G']
        self.distribution = []


        self.net_ET = network.define_et(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_ES = network.define_es(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G = network.define_de(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_D = network.define_dis_g(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_cross_attenstion = network.define_attn(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_fuse_s = network.define_fuse_s(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_fuse_t = network.define_fuse_t(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G2 = network.define_G2(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.lossNet = network.VGG16FeatureExtractor()
        self.lossNet.cuda(opt.gpu_ids[0])


        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
                                                                filter(lambda p: p.requires_grad, self.net_ET.parameters()),
                                                                filter(lambda p: p.requires_grad, self.net_cross_attenstion.parameters()),
                                                                filter(lambda p: p.requires_grad, self.net_fuse_s.parameters()),
                                                                filter(lambda p: p.requires_grad,self.net_fuse_t.parameters()),
                                                                filter(lambda p: p.requires_grad, self.net_G2.parameters()),
                                                                filter(lambda p: p.requires_grad, self.net_ES.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']


        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = (1 - self.mask) * self.img_truth + self.mask

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)



    def test(self):
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')

        # encoder process
        self.image = torch.cat([self.img_m, self.mask], dim=1)
        s_x, s_feature = self.net_ES(self.image)
        t_x, t_feature = self.net_ET(self.image)

        fuse_s = self.net_fuse_s(s_feature)
        fuse_t = self.net_fuse_t(t_feature)

        distribution = self.net_cross_attenstion(s_feature[-1], t_feature[-1])
        mu, sigma = torch.split(distribution, 256, dim=1)
        distribution_normal = torch.distributions.Normal(mu, F.softplus(sigma))

        # decoder process
        for i in range(self.opt.nsampling):
            z = distribution_normal.rsample()
            self.img_g = self.net_G(z, fuse_s, fuse_t)
            self.merged = self.mask * self.img_g[-1].detach() + (1 - self.mask) * self.img_m
            self.img_out = self.net_G2(self.merged, self.mask)
            self.score = self.net_D(self.img_out)
            self.save_results(self.img_out, i, data_name='out')

    def get_distribution(self, distributions):
        # get distribution
        q_distribution, kl = 0, 0
        self.distribution = []
        for distribution in distributions:
            q_mu, q_sigma = distribution
            m_distribution = torch.distributions.Normal(torch.zeros_like(q_mu), torch.ones_like(q_sigma))
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            # kl divergence
            kl += torch.distributions.kl_divergence(q_distribution, m_distribution)
            self.distribution.append([torch.zeros_like(q_mu), torch.ones_like(q_sigma), q_mu, q_sigma])

        return kl

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        self.image = torch.cat([self.img_m, self.mask], dim=1)
        s_x, s_feature = self.net_ES(self.image)
        t_x, t_feature = self.net_ET(self.image)

        fuse_s = self.net_fuse_s(s_feature)
        fuse_t = self.net_fuse_t(t_feature)

        self.kl_g_s = self.get_distribution(s_x)
        self.kl_g_t = self.get_distribution(t_x)
        distribution = self.net_cross_attenstion(s_feature[-1], t_feature[-1])
        mu, sigma = torch.split(distribution, 256, dim=1)
        distribution_normal = torch.distributions.Normal(mu, F.softplus(sigma))
        z = distribution_normal.rsample()

        # decoder process
        results = self.net_G(z, fuse_s, fuse_t)
        self.img_g = []
        for result in results:
            img_g = result
            self.img_g.append(img_g)
        self.img_out = self.img_g[-1].detach()
        self.merged_image = self.img_truth * (1 - self.mask) + self.img_out * self.mask
        self.img_out2 = self.net_G2(self.merged_image, self.mask)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # global
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_img_dg = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""

        # encoder kl loss
        self.loss_kl_s = self.kl_g_s.mean() * self.opt.lambda_kl * self.opt.output_scale
        self.loss_kl_t = self.kl_g_t.mean() * self.opt.lambda_kl * self.opt.output_scale

        # generator adversarial loss
        base_function._freeze(self.net_D)
        D_fake_g = self.net_D(self.img_g[-1])
        D_real_g = self.net_D(self.img_truth)
        self.loss_ad_l = self.L2loss(D_fake_g, D_real_g) * self.opt.lambda_g

        # calculate l1 loss for multi-scale outputs
        loss_app_hole,loss_app_context = 0,0
        for i, (img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_g, self.scale_img, self.scale_mask)):
            loss_app_hole += self.L1loss(img_fake_i*mask_i, img_real_i*mask_i)
            loss_app_context += self.L1loss(img_fake_i * (1-mask_i), img_real_i * (1-mask_i))

        self.loss_app_G1 = loss_app_hole * self.opt.lambda_rec + loss_app_context * self.opt.lambda_rec

        loss_app_hole2 = self.L1loss(self.img_out2 * self.mask, self.img_truth * self.mask)
        loss_app_context2 = self.L1loss(self.img_out2 * (1 - self.mask), self.img_truth * (1 - self.mask))
        self.loss_app_G2 = loss_app_hole2 * self.opt.lambda_rec + loss_app_context2 * self.opt.lambda_rec

        real_feats2 = self.lossNet(self.img_truth)
        fake_feats2 = self.lossNet(self.img_out2)
        comp_feats2 = self.lossNet(self.merged_image)

        self.loss_G_style = base_function.style_loss(real_feats2, fake_feats2) + base_function.style_loss(real_feats2, comp_feats2)
        self.loss_G_content = base_function.perceptual_loss(real_feats2, fake_feats2) + base_function.perceptual_loss(real_feats2, comp_feats2)
        self.loss_G = 0.05*self.loss_G_content + 120*self.loss_G_style

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_dg' and name != 'img_dl':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
