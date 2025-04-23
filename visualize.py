import matplotlib.pyplot as plt

def plot_signals(real_signal, fake_signal):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(real_signal.cpu().numpy())
    axs[0].set_title('Real Sensor Data')
    axs[1].plot(fake_signal.detach().cpu().numpy())
    axs[1].set_title('Generated Sensor Data')
    plt.tight_layout()
    return fig