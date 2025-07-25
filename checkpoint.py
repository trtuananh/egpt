import torch
import os

__all__ = ["save_checkpoint", "load_checkpoint"]


# --- Hàm lưu checkpoint ---
def save_checkpoint(path, model, optimizer, model_args, iter_num, best_val_loss, config_dict):
    """
    Lưu trạng thái của mô hình, optimizer, scheduler, và các thông tin huấn luyện khác.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config_dict,  # save the config dict
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} (Step: {iter_num}, Best Eval Loss: {best_val_loss:.4f})")


# --- Hàm tải checkpoint ---
def load_checkpoint(path, model=None, optimizer=None, device='cpu', weights_only=False):
    """
    Tải trạng thái từ checkpoint để tiếp tục huấn luyện.
    """
    
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    if model is not None and checkpoint['model'] is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    torch.set_rng_state(checkpoint['rng_state'].cpu())
    if torch.cuda.is_available() and checkpoint['cuda_rng_state']:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    print(f"Checkpoint loaded from {path}. Resuming from Step {checkpoint['iter_num']}.")
    return checkpoint
