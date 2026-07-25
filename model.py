import numpy as np
class CustomLSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim        
        concat_dim = hidden_dim + input_dim        
        self.Wf = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.Wi = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.Wc = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.Wo = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))        
        self.Wy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.by = np.zeros((output_dim, 1))
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    def forward_step(self, x_t, h_prev, C_prev):
        index_dim = self.input_dim        
        concat = np.vstack((h_prev, x_t.reshape(-index_dim, 1)))
        f_t = self._sigmoid(np.dot(self.Wf, concat) + self.bf)
        i_t = self._sigmoid(np.dot(self.Wi, concat) + self.bi)
        C_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
        C_t = f_t * C_prev + i_t * C_tilde
        o_t = self._sigmoid(np.dot(self.Wo, concat) + self.bo)
        h_t = o_t * np.tanh(C_t)
        return h_t, C_t
    def forward_sequence(self, sequence: np.ndarray) -> np.ndarray:
        T = sequence.shape[0]
        h = np.zeros((self.hidden_dim, 1))
        C = np.zeros((self.hidden_dim, 1))        
        for t in range(T):
            h, C = self.forward_step(sequence[t], h, C)
        y_pred = np.dot(self.Wy, h) + self.by
        return y_pred.flatten()
    def compute_loss_with_tikhonov(self, X_batch, y_batch, lambda_val):
        loss_mse = 0.0
        gradients = {name: np.zeros_like(param) for name, param in self.__dict__.items() if isinstance(param, np.ndarray)}        
        for X_seq, y_true in zip(X_batch, y_batch):
            y_pred = self.forward_sequence(X_seq)
            loss_mse += np.sum((y_pred - y_true) ** 2)
        loss_mse /= len(X_batch)        
        l2_reg = 0.5 * lambda_val * sum(np.sum(w ** 2) for w in [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy])
        total_loss = loss_mse + l2_reg
        return total_loss