import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    
    # Clipping xác suất dự đoán
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Công thức chuẩn cho Binary Log Loss
    losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Trả về dưới dạng list các số thực
    return losses.tolist()