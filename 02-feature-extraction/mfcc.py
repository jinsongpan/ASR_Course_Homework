import librosa
import numpy as np
# from scipy.fftpack import dct
import pdb

# If you want to see the spectrogram picture
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectrogram(spec, note, file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


# preemphasis config
alpha = 0.97

# Enframe config
frame_len = 400  # 25ms, fs=16kHz
frame_shift = 160  # 10ms, fs=16kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)


# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], [signal[1:] - coeff * signal[:-1]])


def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """

    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win

    return frames


def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:, 0:valid_len])
    return spectrum  # num_frames*valid_len


def fbank(spectrum, num_filter=num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPERATION AFTER MEL FILTER!
    """
    feats = np.zeros((int(fft_len / 2 + 1), num_filter))
    """
        FINISH by YOURSELF
    """

    # step1: 获得滤波器数目在梅尔刻度上的中心频率
    low_mel_freq = 0.0
    high_mel_freq = 2595 * np.log10(1 + (fs / 2) / 700)
    mel_points = np.linspace(low_mel_freq, high_mel_freq, num_filter + 2)
    # step2: 获得对应FFT单元的中心频率
    fft_points = 700 * (10 ** (mel_points / 2595) - 1)
    # # step3: 计算音频帧的能量谱
    # pow_frames = ((1.0 / fft_len) * (spectrum ** 2))
    # step3: 计算mel滤波器组
    bin = (fft_points / (fs / 2)) * (fft_len / 2)
    for i in range(1, num_filter + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            feats[j + 1, i - 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for k in range(center, right):
            feats[j + 1, i - 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    # step4: 计算fbank值并取log
    feats = np.dot(spectrum, feats)
    feats = 20 * np.log10(feats)
    # pdb.set_trace()
    return feats


def mfcc(fbank, num_mfcc=num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    # feats = np.zeros((fbank.shape[0], num_mfcc))
    """
        FINISH by YOURSELF
    """

    def selfdct(x, axis=1, norm="ortho"):
        """scipy.pack.fft.dct()中离散余弦变换的python实现"""
        y = np.zeros(x.shape, dtype=float)
        if axis == 0:
            N = y.shape[0]
            for i in range(y.shape[1]):
                for j in range(y.shape[0]):
                    for k in range(y.shape[0]):
                        y[j][i] += x[k][i] * np.cos(np.pi * j * (2 * k + 1) / (2 * N))
                if norm == "ortho":
                    y[0, i] = 2 * y[0, i] / np.sqrt(4 * N)
                    y[:1, i] = 2 * y[:1, i] / np.sqrt(2 * N)
                else:
                    y[:, i] = 2 * y[:, i]

        elif axis == 1:
            N = y.shape[1]
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    for k in range(y.shape[1]):
                        y[i][j] += x[i][k] * np.cos(np.pi * j * (2 * k + 1) / (2 * N))
                if norm == "ortho":
                    y[i, 0] = 2 * y[i, 0] / np.sqrt(4 * N)
                    y[i, 1:] = 2 * y[i, 1:] / np.sqrt(2 * N)
                else:
                    y[i, :] = 2 * y[i, :]
        else:
            raise ValueError("需要指定dct计算的维度，axis=0 or 1 ...")

        return y

    feats = selfdct(fbank, axis=1, norm='ortho')[:, 1:num_mfcc+1]

    return feats


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)

    # print("fbank_feats", fbank_feats[0])
    # print("mfcc_feats", mfcc_feats[0])

    plot_spectrogram(fbank_feats, 'Filter Bank', 'fbank.png')
    write_file(fbank_feats, './test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC', 'mfcc.png')
    write_file(mfcc_feats, './test.mfcc')


if __name__ == '__main__':
    main()
