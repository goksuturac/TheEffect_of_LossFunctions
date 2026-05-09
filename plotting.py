import matplotlib.pyplot as plt


def plot_history(history, fname):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_score'], label='Training Dice')
    plt.plot(history.history['val_dice_score'], label='Validation Dice')

    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f'{fname}.png')
    plt.close()