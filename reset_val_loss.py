import checkpoint as cp
import torch
import os


def reset_val_loss(ckpt_path, best_val_loss=float('inf')):
    """
    Reset the validation loss in the checkpoint file.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(checkpoint.keys(), checkpoint['best_val_loss'])
    
    # Reset the best validation loss
    print(f"Resetting validation loss in {ckpt_path} from {checkpoint["best_val_loss"]} to {best_val_loss}.")
    checkpoint['best_val_loss'] = best_val_loss
    
    torch.save(checkpoint, ckpt_path)

    print(f"Validation loss reset in {ckpt_path}.")

if __name__ == "__main__":
    ckpt_path = 'out/egpt_long/best_ckpt.pt'
    reset_val_loss(ckpt_path)
