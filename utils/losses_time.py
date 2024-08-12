import torch

from models.model_GDSS_time import get_timestep_embedding
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise
from utils.sde import VESDE, VPSDE, subVPSDE

def get_score_fn(sde, model, train=True, continuous=True):
    if not train:
        model.eval()
    model_fn = model

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, adj, flags, temb, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous:
                score = model_fn(x, adj, flags, temb)
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, adj, flags, temb, t):
            if continuous:
                score = model_fn(x, adj, flags, temb)
            else:  
                raise NotImplementedError(f"Discrete not supported")
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

    return score_fn

def get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=False, continuous=True, 
                    likelihood_weighting=False, eps=1e-5):
  
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  
    def loss_fn(model_x, model_adj, model_temb, x, adj, flags):
        score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=train, continuous=continuous)

        ### 后面eps作用：限定t不为0和1
        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
        #flags = node_flags(x)
        temb = model_temb(t)
        
        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=True) 
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, temb, t)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, temb, t)

        if not likelihood_weighting:
            losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)

            losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        else:
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj

        return torch.mean(losses_x), torch.mean(losses_adj)

    return loss_fn

def load_loss_fn(args, train=True):
    reduce_mean = args.reduce_mean
    sde_x = VPSDE(beta_min=args.beta_min_x, beta_max=args.beta_max_x, N=args.num_scales_x)
    sde_adj = VESDE(sigma_min=args.beta_min_adj, sigma_max=args.beta_max_adj, N=args.num_scales_adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=args.eps)
    
    return loss_fn

if __name__ == '__main__':
    #def main():
    warnings.filterwarnings(action='ignore')
    
    min_loss = 1e10
    
    
    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        
        ### train
        loss_x_train, loss_adj_train = train(epoch, train_loader)
        
        ### val
        loss_x_val, loss_adj_val = test(test_loader)
        total_loss = loss_x_val + loss_adj_val
        
        
        print('=============================================================')
        print('train',
            'Epoch: {:04d}'.format(epoch),
            'loss_x_train: {:.6f}'.format(loss_x_train),
            'loss_adj_train: {:.6f}'.format(loss_adj_train))  
        
        print('val',
            'Epoch: {:04d}'.format(epoch),
            'loss_x_val: {:.6f}'.format(loss_x_val),
            'loss_adj_val: {:.6f}'.format(loss_adj_val))       
        print('=============================================================')
    
        ### save train.csv
        if os.path.exists(log_path_train) == False:
            with open(log_path_train, 'w', newline='') as train_writer_csv:
                header_list = ['epoch', 'loss_x', 'loss_adj']
                train_writer = csv.DictWriter(train_writer_csv, fieldnames=header_list)
                train_writer.writeheader()
        with open(log_path_train, 'a', newline='') as train_writer_csv:
            train_writer = csv.writer(train_writer_csv)
            train_writer.writerow([epoch, str(loss_x_train), str(loss_adj_train)])
        
        ### sava val.csv 
        if os.path.exists(log_path_val) == False:
            with open(log_path_val, 'w', newline='') as val_writer_csv:
                header_list = ['epoch', 'loss_x', 'loss_adj']
                val_writer = csv.DictWriter(val_writer_csv, fieldnames=header_list)
                val_writer.writeheader()
        with open(log_path_val, 'a', newline='') as val_writer_csv:
            val_writer = csv.writer(val_writer_csv)
            val_writer.writerow([epoch, str(loss_x_val), str(loss_adj_val)])
        
        state = {'model_x': model_x.state_dict(),
                'model_adj': model_adj.state_dict(),
                'model_t': model_t.state_dict(),
                'ema_x': ema_x.state_dict(),
                'ema_adj': ema_adj.state_dict(),
                'loss': total_loss,
                'epoch': epoch,
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_adj': optimizer_adj.state_dict()}
                          
        ### save checkpoint
        torch.save(state, checkpoint_current)     
        
        if total_loss < min_loss :
            print('Saving..')
            torch.save(state, checkpoint_best)
            min_loss = total_loss   