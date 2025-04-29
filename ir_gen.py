import torch

def generate_synthetic_ir_batch(
    batch_size,
    duration=2.0,
    sample_rate=24000,
    device="cpu",
    max_reflections=10
):
    
    n_samples = int(duration * sample_rate)
    t = torch.arange(n_samples, device=device).unsqueeze(0);import random as rand
    #return torch.sin(10 * t * (rand.random() * 4.5 + 0.5) / float(n_samples))

    pre_delay_ms = torch.randint(0, 20, (batch_size,), device=device)
    pre_delay_samples = (pre_delay_ms * sample_rate / 1000.0).long()

    decay_factors = torch.rand(batch_size, device=device) * 1.5 + 0.5
    spectral_tilts = torch.rand(batch_size, device=device) * 6.0 - 3.0

    early_times = torch.rand(batch_size, max_reflections, device=device) * 0.095 + 0.005
    early_amps = torch.rand(batch_size, max_reflections, device=device) * 0.7 + 0.3
    early_amps *= torch.exp(-early_times * 10)

    num_reflections = torch.randint(1, max_reflections + 1, (batch_size,), device=device)
    mask = torch.arange(max_reflections, device=device).unsqueeze(0) < num_reflections.unsqueeze(1)
    early_amps *= mask

    early_idxs = (early_times * sample_rate).long() + pre_delay_samples.unsqueeze(1)
    ir_batch = torch.zeros((batch_size, n_samples), device=device)
    for b in range(batch_size):
        valid = early_idxs[b] < n_samples
        ir_batch[b, early_idxs[b, valid]] += early_amps[b, valid]

    max_early_time = early_times.max(dim=1).values
    tail_start = pre_delay_samples + (max_early_time * sample_rate).long()
    tail_lengths = n_samples - tail_start

    max_tail = tail_lengths.max()
    t_tail = torch.arange(max_tail, device=device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = t_tail < tail_lengths.unsqueeze(1)

    noise = torch.randn((batch_size, max_tail), device=device)
    tail = noise * valid_mask

    for b in range(batch_size):
        end = tail_start[b] + tail_lengths[b]
        ir_batch[b, tail_start[b]:end] += tail[b, :tail_lengths[b]]

    low_pass = torch.rand(batch_size, device=device) * 9500.0 + 500.0 # [500, 10000]
    high_pass = torch.rand(batch_size, device=device) * 100.0 # [0, 100]

    freqs = torch.fft.rfftfreq(n_samples, 1.0 / sample_rate).to(device).unsqueeze(0)
    low_pass_mask = (freqs <= low_pass.unsqueeze(1)).float()
    high_pass_mask = (freqs >= high_pass.unsqueeze(1)).float()
    combined_mask = low_pass_mask * high_pass_mask

    spectrum = torch.fft.rfft(ir_batch, n=n_samples)
    spectrum *= combined_mask.to(dtype=spectrum.dtype)
    ir_batch = torch.fft.irfft(spectrum, n=n_samples)

    decay = torch.exp(-t_tail / ((decay_factors * sample_rate / 6.91).unsqueeze(1)))
    mod = 1 + 0.1 * torch.sin(2 * torch.pi * 0.5 * t_tail / sample_rate)

    tail = decay * mod
    tail *= valid_mask

    for b in range(batch_size):
        end = tail_start[b] + tail_lengths[b]
        ir_batch[b, tail_start[b]:end] *= tail[b, :tail_lengths[b]]

    return ir_batch

def test(ir_generator):

    import numpy as np
    import sounddevice as sd
    import scipy.signal
    import matplotlib.pyplot as plt 

    def play_sine_with_and_without_reverb(ir, sample_rate=24000, duration=2.0):
        # Generate a short sine pulse at A440 (440 Hz)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave = np.sin(2 * np.pi * 440 * t)
        
        # Create a very short pulse (first few ms only)
        pulse_duration = 0.1  # 100 ms
        pulse_samples = int(pulse_duration * sample_rate)
        sine_wave[pulse_samples:] = 0  # zero out everything after the pulse
        
        # Normalize sine wave
        sine_wave /= np.max(np.abs(sine_wave))
        
        # Convolve with IR
        convolved = scipy.signal.fftconvolve(sine_wave, ir, mode='same')
        convolved /= np.max(np.abs(convolved))  # normalize
        
        print("Playing dry sine pulse, press enter to continue...")
        sd.play(sine_wave, samplerate=sample_rate)
        sd.wait()
        input("")

        print("Playing sine pulse with reverb...")
        sd.play(convolved, samplerate=sample_rate)
        sd.wait()

    while True:

        x = ir_generator()

        #plt.plot(x)
        #plt.show()

        play_sine_with_and_without_reverb(x)
        if input() == 'q': break

if __name__ == "__main__":

    option = input("do the speed test or do the listening test? (1/2): ")

    if option == '1':

        batch_size = 32
        sample_rate = 24000

        import time 

        start = time.time()
        for _ in range(100): x = generate_synthetic_ir_batch(batch_size, sample_rate=sample_rate, device='cpu')
        end = time.time()

        batches_per_second = int(100.0 / (end - start))
        print(f'Generated an avg of {batches_per_second} batches per second using CUDA and batch size 32')
        print(f'ir shape = {x.shape}')

    elif option == '2':

        def g():

            x = generate_synthetic_ir_batch(1)[0]
            return x

        test(g)

    else: print('invalid option')