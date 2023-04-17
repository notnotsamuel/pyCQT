import numpy as np
from scipy.sparse import hstack, vstack, coo_matrix

class CQT:
    """
    Class for performing the Constant Q Transform (CQT) on input signals.
    """
    def __init__(self, fmin, fmax, bins, fs, window):
        """
        Initialize the CQT object.
        
        :param fmin: Minimum frequency.
        :param fmax: Maximum frequency.
        :param bins: Number of bins per octave.
        :param fs: Sampling frequency.
        :param window: Window function to apply on the input signal.
        """
        self.fmin = fmin # Minimum frequency of interest
        self.fmax = fmax # Maximum frequency of interest
        self.bins = bins # Number of frequency bins per octave
        self.window = window # Window function to apply on the input signal
        self.fs = fs  # Sampling frequency of the input signal
        self.quality_factor = 1 / (pow(2, 1.0 / bins) - 1) # Quality factor (Q), which represents the filter bandwidth
        K = int(np.ceil(bins * np.log2(fmax / fmin))) # Total number of frequency bins (K) for the given frequency range


        # Calculate the length of the FFT
        self.fftlen = int(2 ** np.ceil(np.log2(self.quality_factor * fs / fmin)))

        # Create the filter bank (kernel) for the CQT
        self.ker = []
        for k in range(K, 0, -1):
            N = np.ceil(self.quality_factor * fs / (fmin * 2 ** ((k - 1.0) / bins)))
            tmpKer = window(N) * np.exp(2 * np.pi * 1j * self.quality_factor * np.arange(N) / N) / N
            ker = np.fft.fft(tmpKer, self.fftlen)
            self.ker += [coo_matrix(ker, dtype=np.complex128)]

        # Process the filter bank for CQT calculation
        self.ker.reverse()
        self.ker = vstack(self.ker).tocsc().transpose().conj() / self.fftlen

    def compute_cqt_fast(self, x):
        """
        Perform the CQT transform on the input signal using a faster method.
        
        :param x: Input signal.
        :return: CQT coefficients.
        """
        return (np.fft.fft(x, self.fftlen).reshape(1, self.fftlen) * self.ker)[0]

    def compute_cqt_bruteforce(self, x):
        """
        Perform the CQT transform on the input signal using a slower, brute-force method.
        
        :param x: Input signal.
        :return: CQT coefficients.
        """
        cq = []
        for k in range(1, int(np.ceil(self.bins * np.log2(self.fmax / self.fmin))) + 1):
            N = np.ceil(self.quality_factor * self.fs / (self.fmin * 2 ** ((k - 1.0) / self.bins)))
            cq += [x[:N].dot(self.window(N) * np.exp(-2 * np.pi * 1j * self.quality_factor * np.arange(N) / N) / N)]
        return np.array(cq)

    @staticmethod
    def hamming(length):
        """
        Generate a Hamming window.
        
        :param length: Length of the window.
        :return: Hamming window.
        """
        return 0.46 - 0.54 * np.cos(2 * np.pi * np.arange(length) / length)
    
    @staticmethod
    def hanning(length):
        """
        Generate a Hanning window.
        
        :param length: Length of the window.
        :return: Hanning window.
        """
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / length))
    
    @staticmethod
    def blackman(length):
        """
        Generate a Blackman window.
        
        :param length: Length of the window.
        :return: Blackman window.
        """
        return 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length) + 0.08 * np.cos(4 * np.pi * np.arange(length) / length)



