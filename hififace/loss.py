import torch
from lib.loss import Loss, LossInterface


class HifiFaceLoss(LossInterface):
    def __init__(self, args):
        super().__init__(args)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((64, 64)).to("cuda").eval()

    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(G_dict["d_adv"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Shape loss
        if self.args.W_shape:
            L_shape = Loss.get_L1_loss(G_dict["q_fuse"], G_dict["q_swapped_high"]) 
            L_shape += Loss.get_L1_loss(G_dict["q_fuse"], G_dict["q_swapped_low"])
            L_G += self.args.W_shape * L_shape/68
            self.loss_dict["L_shape"] = round(L_shape.item(), 4)

        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped_high"])
            L_id += Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped_low"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(G_dict["I_swapped_high"], G_dict["I_target"], G_dict["same_person"], self.args.batch_per_gpu)
            L_recon += Loss.get_L1_loss_with_same_person(G_dict["I_swapped_low"], G_dict["I_target"], G_dict["same_person"], self.args.batch_per_gpu)
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)

        # Cycle loss
        if self.args.W_cycle:
            L_cycle = Loss.get_L1_loss(G_dict["I_target"], G_dict["I_cycle"])
            L_G += self.args.W_cycle * L_cycle
            self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)

        # LPIPS loss
        if self.args.W_lpips:
            # L_lpips = Loss.get_lpips_loss(G_dict["I_cycle"], G_dict["I_target"])
            L_lpips = Loss.get_lpips_loss(G_dict["I_swapped_high"], G_dict["I_target"])
            L_lpips += Loss.get_lpips_loss(G_dict["I_swapped_low"], G_dict["I_target"])
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, D_dict):
        L_true = Loss.get_BCE_loss(D_dict["d_true"], True)
        L_fake = Loss.get_BCE_loss(D_dict["d_fake"], False)
        L_reg = Loss.get_r1_reg(D_dict["d_true"], D_dict["I_target"])
        L_D = L_true + L_fake + L_reg
        
        self.loss_dict["L_D"] = round(L_D.item(), 4)
        self.loss_dict["L_true"] = round(L_true.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)

        return L_D
