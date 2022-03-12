import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from submodel import arcface
from submodel.deep3dmm import ParametricFaceModel, ReconNet
from submodel.faceparser import BiSeNet
from lib.utils import AdaIN, weight_init
from lib.blocks import ResBlock, AdaINResBlock


class HififaceGenerator(nn.Module):
    def __init__(self):
        super(HififaceGenerator, self).__init__()
        
        self.SAIE = ShapeAwareIdentityExtractor()
        self.SFFM = SemanticFacialFusionModule()
        self.E = Encoder()
        self.D = Decoder()

    def forward(self, I_s, I_t):
        
        # 3D Shape-Aware Identity Extractor
        v_sid, coeff_dict_fuse = self.SAIE(I_s, I_t)
        
        # Encoder
        z_latent, z_enc = self.E(I_t)

        # Decoder
        z_dec = self.D(z_latent, v_sid)

        # Semantic Facial Fusion Module
        I_swapped_high, I_swapped_low, z_fuse = self.SFFM(z_enc, z_dec, v_sid, I_t)
        
        return I_swapped_high, I_swapped_low, z_fuse, coeff_dict_fuse



class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self):
        super(ShapeAwareIdentityExtractor, self).__init__()

        # face recognition model: arcface
        self.F_id = arcface.Backbone(50, 0.6, 'ir_se').eval()
        self.F_id.load_state_dict(torch.load('ptnn/arcface.pth', map_location="cuda"), strict=False)
        self.F_id.eval()
        for param in self.F_id.parameters():
            param.requires_grad = False

        # 3D face reconstruction model
        self.net_recon = ReconNet()
        state_dict = torch.load("ptnn/deep3d.pth", map_location="cuda")
        self.net_recon.load_state_dict(state_dict['net_recon'])
        self.net_recon.eval()
        for param in self.net_recon.parameters():
            param.requires_grad = False

        self.facemodel = ParametricFaceModel(is_train=False)
        
    def forward(self, I_s, I_t):
        # id of Is
        with torch.no_grad():
            v_id = self.get_id(I_s)

            # 3d params of Is
            coeff_dict_s = self.get_coeff3d(I_s)

            # 3d params of It
            coeff_dict_t = self.get_coeff3d(I_t)

        # fused 3d parms
        coeff_dict_fuse = coeff_dict_t.copy()
        coeff_dict_fuse["id"] = coeff_dict_s["id"]

        # concat all to obtain the 3D shape-aware identity(v_sid)
        v_sid = torch.cat([v_id, coeff_dict_fuse["id"], coeff_dict_fuse["exp"], coeff_dict_fuse["angle"]], dim=1)
        
        return v_sid, coeff_dict_fuse

    def get_id(self, I):
        v_id = self.F_id(F.interpolate(I[:, :, 16:240, 16:240], [112, 112], mode='bilinear', align_corners=True))
        return v_id

    def get_coeff3d(self, I):
        coeffs = self.net_recon(I[:, :, 16:240, 16:240]*0.5+0.5)
        coeff_dict = self.facemodel.split_coeff(coeffs)

        return coeff_dict

    def get_lm3d(self, coeff_dict):
        
        # get 68 3d landmarks
        face_shape = self.facemodel.compute_shape(coeff_dict['id'], coeff_dict['exp'])
        rotation = self.facemodel.compute_rotation(coeff_dict['angle'])

        face_shape_transformed = self.facemodel.transform(face_shape, rotation, coeff_dict['trans'])
        face_vertex = self.facemodel.to_camera(face_shape_transformed)
        
        face_proj = self.facemodel.to_image(face_vertex)
        lm3d = self.facemodel.get_landmarks(face_proj)

        return lm3d


class SemanticFacialFusionModule(nn.Module):
    def __init__(self, norm='in', activation='lrelu', styledim=659):
        super(SemanticFacialFusionModule, self).__init__()

        self.ResBlock = ResBlock(256, 256, scale_factor=1, norm=norm, activation=activation)
        self.AdaINResBlock = AdaINResBlock(256, 259, scale_factor=1, activation=activation, style_dim=styledim)
        
        self.F_up = F_up()
        self.face_pool = nn.AdaptiveAvgPool2d((64, 64)).eval()

        # face Segmentation model: HRNet [Sun et al., 2019]
        self.segmentation_net = BiSeNet(n_classes=19).to('cuda')
        self.segmentation_net.load_state_dict(torch.load('ptnn/faceparser.pth', map_location="cuda"))
        self.segmentation_net.eval()
        for param in self.segmentation_net.parameters():
            param.requires_grad = False
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))

    def get_mask(self, I):
        with torch.no_grad():
            size = I.size()[-1]
            parsing = self.segmentation_net(F.interpolate(I, size=(512,512), mode='bilinear', align_corners=True)).max(1)[1]
            mask = torch.where(parsing>0, 1, 0)
            mask-= torch.where(parsing>13, 1, 0)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(size,size), mode='nearest')
            mask = self.blur(mask)
        return mask

    def forward(self, z_enc, z_dec, v_sid, I_target):

        M_high = self.get_mask(I_target).detach()
        M_low = self.face_pool(M_high)

        # z_enc 256 64 64
        # z_dec 256 64 64
        
        # z_fuse 256 64 64
        z_enc_ = self.ResBlock(z_enc)
        z_fuse = z_dec * M_low.repeat(1, 256, 1, 1) + z_enc_ * (1-M_low.repeat(1, 256, 1, 1))
        I_out_low = self.AdaINResBlock(z_fuse, v_sid)

        # I_low 3 64 64
        I_swapped_low = I_out_low[:, :3, :, :] * M_low.repeat(1, 3, 1, 1) + self.face_pool(I_target) * (1-M_low.repeat(1, 3, 1, 1))

        # I_out_high 3 256 256
        I_out_high = self.F_up(I_out_low[:, 3:, :, :])

        # I_r 3 256 256
        I_swapped_high = I_out_high * M_high.repeat(1, 3, 1, 1) + I_target * (1-M_high.repeat(1, 3, 1, 1))
        
        return I_swapped_high, I_swapped_low, z_fuse


class F_up(nn.Module):
    def __init__(self, norm='in', activation='lrelu'):
        super(F_up, self).__init__()
        self.ResBlock_image1 = ResBlock(256, 256, scale_factor=2, norm=norm, activation=activation)
        self.ResBlock_image2 = ResBlock(256, 256, scale_factor=2, norm=norm, activation=activation)
        self.ResBlock_image3 = ResBlock(256, 64, scale_factor=1, norm=norm, activation=activation)
        self.LastConv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.Tanh = nn.Tanh()

    def forward(self, I_out_low):
        feat_image1 = self.ResBlock_image1(I_out_low)
        feat_image2 = self.ResBlock_image2(feat_image1)
        feat_image3 = self.ResBlock_image3(feat_image2)
        out = self.LastConv(feat_image3)

        return self.Tanh(out)


class Encoder(nn.Module):
    def __init__(self, norm='in', activation='lrelu'):
        super(Encoder, self).__init__()

        self.InitConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.ResBlock1 = ResBlock(64, 128, scale_factor=0.5, norm=norm, activation=activation)
        self.ResBlock2 = ResBlock(128, 256, scale_factor=0.5, norm=norm, activation=activation)
        self.ResBlock3 = ResBlock(256, 512, scale_factor=0.5, norm=norm, activation=activation)
        self.ResBlock4 = ResBlock(512, 512, scale_factor=0.5, norm=norm, activation=activation)
        self.ResBlock5 = ResBlock(512, 512, scale_factor=0.5, norm=norm, activation=activation)
        self.ResBlock6 = ResBlock(512, 512, scale_factor=1, norm=norm, activation=activation)
        self.ResBlock7 = ResBlock(512, 512, scale_factor=1, norm=norm, activation=activation)

        self.apply(weight_init)

    def forward(self, It):
        feat0 = self.InitConv(It) # 32x128x128
        feat1 = self.ResBlock1(feat0) # 32x128x128
        feat2 = self.ResBlock2(feat1) # 64x64x64
        feat3 = self.ResBlock3(feat2) # 128x32x32
        feat4 = self.ResBlock4(feat3) # 256x16xx16
        feat5 = self.ResBlock5(feat4) # 512x8x8
        feat6 = self.ResBlock6(feat5) # 1024x4x4
        feat7 = self.ResBlock7(feat6) # 1024x4x4

        return feat7, feat2

class Decoder(nn.Module):
    def __init__(self, activation='lrelu', styledim=659):
        super(Decoder, self).__init__()

        self.InitConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.AdaINResBlock1 = AdaINResBlock(512, 512, scale_factor=1, activation=activation, style_dim=styledim)
        self.AdaINResBlock2 = AdaINResBlock(512, 512, scale_factor=1, activation=activation, style_dim=styledim)
        self.AdaINResBlock3 = AdaINResBlock(512, 512, scale_factor=2, activation=activation, style_dim=styledim)
        self.AdaINResBlock4 = AdaINResBlock(512, 512, scale_factor=2, activation=activation, style_dim=styledim)
        self.AdaINResBlock5 = AdaINResBlock(512, 256, scale_factor=2, activation=activation, style_dim=styledim)

        self.apply(weight_init)

    def forward(self, feat, v_sid):
        feat1 = self.AdaINResBlock1(feat, v_sid) # 32x128x128
        feat2 = self.AdaINResBlock2(feat1, v_sid) # 64x64x64
        feat3 = self.AdaINResBlock3(feat2, v_sid) # 128x32x32
        feat4 = self.AdaINResBlock4(feat3, v_sid) # 256x16xx16
        z_dec = self.AdaINResBlock5(feat4, v_sid) # 512x8x8

        return z_dec
