import numpy as np
import pytest
from column_buckling import find_critical_load

def test_known_case():
    # נתונים עבור עמוד פלדה ספציפי שחישבת מראש
    L, E, A, r, c, e, sigma_allow = 3000, 200000, 5000, 50, 100, 20, 250
    result = find_critical_load(L, E, A, r, c, e, sigma_allow)
    
    # בדיקה שהתוצאה לא שלילית והיא הגיונית (למשל, קטנה מעומס אוילר)
    euler_load = (np.pi**2 * E * (A * r**2)) / (L**2)
    assert 0 < result < euler_load
    
    # בדיקה שהתוצאה מאפסת את המשוואה (בקירוב)
    P = result
    sigma_max = (P/A) * (1 + (e*c/r**2) * (1/np.cos((L/(2*r)) * np.sqrt(P/(E*A)))))
    assert np.isclose(sigma_max, sigma_allow, rtol=1e-3)
