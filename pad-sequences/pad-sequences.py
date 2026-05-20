import numpy as np


def pad_sequences(seqs, pad_value=0, max_len=None):
    """Returns: np.ndarray of shape (N, L) where:

    N = len(seqs)
    L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)

    # 1. Xác định độ dài L theo đúng yêu cầu đề bài
    if max_len is not None:
        L = max_len
    else:
        # Sử dụng "or 0" phòng trường hợp danh sách seqs rỗng
        L = max(len(seq) for seq in seqs) if N > 0 else 0

    # 2. Khởi tạo ma trận kết quả phủ kín bằng pad_value
    # Ép kiểu dữ liệu (dtype) dựa trên phần tử đầu tiên nếu có để tránh lỗi ép kiểu mặc định
    dtype = np.array(seqs[0]).dtype if (N > 0 and len(seqs[0]) > 0) else int
    padded_matrix = np.full((N, L), pad_value, dtype=dtype)

    # 3. Điền dữ liệu thực tế vào ma trận
    for i, seq in enumerate(seqs):
        if L == 0:
            continue

        # Lấy độ dài thực tế cần điền (cắt bớt nếu chuỗi dài hơn L)
        actual_len = min(len(seq), L)

        # Ghi đè chuỗi vào hàng thứ i, từ cột 0 đến actual_len
        padded_matrix[i, :actual_len] = seq[:actual_len]

    return padded_matrix


