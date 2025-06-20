import numpy as np

def load_data():
    """
    Returns:
      x (ndarray): shape (m,) input values (Population)
      y (ndarray): shape (m,) target values (Profit)
    """
    x = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 
                  8.3829, 7.4764, 8.5781, 6.4862, 5.0546, 
                  5.7107, 14.164, 5.734, 8.4084, 5.6407, 
                  5.3794, 6.3654, 5.1301, 6.4296, 7.0708,
                  6.1891, 20.27, 5.4901, 6.3261, 5.5649])
    
    y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233, 
                  11.886, 4.3483, 12.0, 6.5987, 3.8166, 
                  3.2522, 15.505, 3.1551, 7.2258, 0.71618, 
                  3.5129, 5.3048, 0.56077, 3.6518, 5.3893,
                  3.1386, 21.767, 4.263, 5.1875, 3.0825])
    
    return x, y
