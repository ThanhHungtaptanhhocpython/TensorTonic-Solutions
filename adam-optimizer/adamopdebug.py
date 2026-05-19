import numpy as np


def adam_step(
    param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
):
    """One Adam optimizer update step.

    Return (param_new, m_new, v_new).
    """
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)

    # 1. Cập nhật ước lượng moment bậc 1 (biến m)
    m_new = beta1 * m + (1 - beta1) * grad

    # 2. Cập nhật ước lượng moment bậc 2 (biến v)
    v_new = beta2 * v + (1 - beta2) * (grad**2)

    # 3. Hiệu chỉnh chệch (Bias Correction)
    # Lưu ý: t phải bắt đầu từ 1 (1-indexed) để tránh mẫu số bằng 0
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    # 4. Cập nhật tham số mới (param_new)
    param_new = param - (lr / (np.sqrt(v_hat) + eps)) * m_hat

    # 5. Trả về đúng bộ 3 giá trị theo yêu cầu đề bài
    return param_new, m_new, v_new


if __name__ == "__main__":
    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, -0.2, 0.3])
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    t = 1

    param_new, m_new, v_new = adam_step(param, grad, m, v, t)

    print("param_new:", param_new)
    print("m_new:", m_new)
    print("v_new:", v_new)


