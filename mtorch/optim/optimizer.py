from mtorch.config import Device


class Optimizer:

    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class Adam(Optimizer):

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [Device.xp.zeros_like(p.data) for p in self.params]
        self.v = [Device.xp.zeros_like(p.data) for p in self.params]

        self.is_cupy = hasattr(Device.xp, "ElementwiseKernel")
        if self.is_cupy:
            self._fused_step = Device.xp.ElementwiseKernel(
                in_params="T grad, T lr, T beta1, T beta2, T eps, T beta1_t, T beta2_t",
                out_params="T param, T m, T v",
                operation="""
                // 1. Update biased first and second moment estimates
                m = beta1 * m + (1.0 - beta1) * grad;
                v = beta2 * v + (1.0 - beta2) * grad * grad;
                
                // 2. Compute bias-corrected momentum
                T m_hat = m / (1.0 - beta1_t);
                T v_hat = v / (1.0 - beta2_t);
                
                // 3. Update the actual weights
                param -= lr * m_hat / (sqrt(v_hat) + eps);
                """,
                name="fused_adam",
            )

    def step(self):
        self.t += 1

        beta1_t = self.beta1**self.t
        beta2_t = self.beta2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.is_cupy:
                self._fused_step(
                    p.grad,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    beta1_t,
                    beta2_t,
                    p.data,
                    self.m[i],
                    self.v[i],
                )
            else:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)

                m_hat = self.m[i] / (1 - beta1_t)
                v_hat = self.v[i] / (1 - beta2_t)

                p.data -= self.lr * m_hat / (Device.xp.sqrt(v_hat) + self.eps)


class SGD(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [Device.xp.zeros_like(p.data) for p in self.params]

    def step(self):

        for p, v in zip(self.params, self.velocities):
            v *= self.momentum
            v += self.lr * p.grad

            p.data -= v
