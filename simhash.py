class Simhash:
    def __init__(self, text, f=64):
        self.f = f
        self.hash = self._compute(text) if text else 0

    def _compute(self, text):
        v = [0] * self.f
        for i in range(len(text)-1):
            gram = text[i:i+2]
            h = hash(gram) & ((1 << self.f) - 1)
            for j in range(self.f):
                if (h >> j) & 1:
                    v[j] += 1
                else:
                    v[j] -= 1
        fingerprint = 0
        for j in range(self.f):
            if v[j] > 0:
                fingerprint |= (1 << j)
        return fingerprint

    def distance(self, other):
        x = self.hash ^ other.hash
        return bin(x).count('1')