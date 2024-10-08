import torch
import numpy as np
import abc
from tqdm import trange

from utils.graph_utils import init_flags, mask_adjs, mask_x, gen_noise
from utils.losses_time import get_score_fn
from utils.sde import VPSDE, VESDE, subVPSDE

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse_temb(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass

class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, obj, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj

    def update_fn(self, x, adj, flags, temb, t):
        dt = -1. / self.rsde.N

        if self.obj=='x':
            z = gen_noise(x, flags, sym=False)
            drift, diffusion = self.rsde.sde(x, adj, flags, temb, t, is_adj=False)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
            return x, x_mean

        elif self.obj=='adj':
            z = gen_noise(adj, flags)
            drift, diffusion = self.rsde.sde(x, adj, flags, temb, t, is_adj=True)
            adj_mean = adj + drift * dt
            adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")

class ReverseDiffusionPredictor(Predictor):
    def __init__(self, obj, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj

    def update_fn(self, x, adj, flags, temb, t):
        if self.obj == 'x':
            f, G = self.rsde.discretize(x, adj, flags, temb, t, is_adj=False)
            z = gen_noise(x, flags, sym=False)
            x_mean = x - f
            x = x_mean + G[:, None, None] * z
            return x, x_mean

        elif self.obj == 'adj':
            f, G = self.rsde.discretize(x, adj, flags, temb, t, is_adj=True)
            z = gen_noise(adj, flags)
            adj_mean = adj - f
            adj = adj_mean + G[:, None, None] * z
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")

class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        self.obj = obj
        pass

    def update_fn(self, x, adj, flags, t):
        if self.obj == 'x':
            return x, x
        elif self.obj == 'adj':
            return adj, adj
        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)
        self.obj = obj

    def update_fn(self, x, adj, flags, temb, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        if self.obj == 'x':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, temb, t)
                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return x, x_mean

        elif self.obj == 'adj':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, temb, t)
                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported")
    
# -------- inpainter sampler plus --------
def get_pc_inpainter_plus(sde_x, sde_adj, predictor='Euler', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda',
                   max_node_num=36, max_feat_num=3):

    def pc_inpainter(model_x, model_adj, model_t, init_flags, part_x, part_adj, flags_x):
        shape_x = (init_flags.shape[0], max_node_num, max_feat_num)
        shape_adj = (init_flags.shape[0], max_node_num, max_node_num)
        
        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

        predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor 
        corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

        predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
        corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

        predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
        corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

        with torch.no_grad():
            # -------- Initial sample --------
            x = sde_x.prior_sampling(shape_x).to(device) 
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
            # mix data and noise
            x = x * (1. - flags_x) + part_x
            adj = adj * (1. - part_adj) + part_adj
            # save num of flags
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)

            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
            
            # -------- Reverse diffusion process --------
            for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                temb = model_t(vec_t)
                
                _x = x
                x, x_mean = corrector_obj_x.update_fn(x, adj, flags, temb, vec_t)
                adj, adj_mean = corrector_obj_adj.update_fn(x, adj, flags, temb, vec_t)
                
                masked_x_mean, std = sde_x.marginal_prob(part_x, vec_t)                
                masked_x = masked_x_mean + torch.randn_like(x) * std[:, None, None]
                x = x * (1. - flags_x) + masked_x * flags_x                                
                x_mean = x_mean * (1. - flags_x) + masked_x_mean * flags_x
                                
                masked_adj_mean, std = sde_adj.marginal_prob(part_adj, vec_t)
                masked_adj = masked_adj_mean + torch.randn_like(adj) * std[:, None, None]
                adj = adj * (1. - part_adj) + masked_adj * part_adj
                adj_mean = adj_mean * (1. - part_adj) + masked_adj_mean * part_adj
                
                _x = x
                x, x_mean = predictor_obj_x.update_fn(x, adj, flags, temb, vec_t)
                adj, adj_mean = predictor_obj_adj.update_fn(x, adj, flags, temb, vec_t)
                
                masked_x_mean, std = sde_x.marginal_prob(part_x, vec_t)                
                masked_x = masked_x_mean + torch.randn_like(x) * std[:, None, None]
                x = x * (1. - flags_x) + masked_x * flags_x                                
                x_mean = x_mean * (1. - flags_x) + masked_x_mean * flags_x                
                
                masked_adj_mean, std = sde_adj.marginal_prob(part_adj, vec_t)
                masked_adj = masked_adj_mean + torch.randn_like(adj) * std[:, None, None]
                adj = adj * (1. - part_adj) + masked_adj * part_adj
                adj_mean = adj_mean * (1. - part_adj) + masked_adj_mean * part_adj            
                
            print(' ')
            return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
    return pc_inpainter  
    
