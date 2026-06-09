import numpy as np
from scipy.optimize import bisect, newton

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    מציאת העומס המקסימלי המותר P בניוטון לפי נוסחת הסקנט.
    """
    # בדיקה דינמית של יחידות: אם r קטן מדי (למשל ס"מ), נמיר אותו למ"מ.
    # בפרופילי HEB/IPE, אם r מוצג בס"מ הוא יהיה קטן (למשל 3.5 במקום 35).
    if r < 15:  
        r = r * 10.0
        c = c * 10.0
        e = e * 10.0

    # פונקציית המטרה: f(P) = sigma_max(P) - sigma_allow
    def f(P):
        if P <= 0:
            return -sigma_allow
        
        # הארגומנט של הקוסינוס (ברדיאנים)
        val = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # חסם פיזיקלי - הקוסינוס לא יכול להתאפס או להיות שלילי בטווח הלחיצה
        if val >= np.pi / 2:
            return float('inf')
            
        # נוסחת הסקנט: sec(x) = 1 / cos(x)
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(val)))
        return sigma_max - sigma_allow

    # חישוב עומס אוילר כחסם עליון מוחלט
    p_euler = (np.pi**2 * E * A * r**2) / L**2
    
    # נקודת התחלה בטוחה לחיפוש (למשל 50% מעומס אוילר)
    p_start = p_euler * 0.5

    try:
        # ניסיון ראשון: שיטת ניוטון-רפסון (מהירה ומדויקת ביותר)
        critical_load = newton(f, x0=p_start, tol=1e-5, maxiter=200)
    except (RuntimeError, ValueError):
        # גיבוי: אם ניוטון נכשל, נבצע סריקה אדפטיבית בשיטת החצייה (Bisect)
        p_min = 1e-5
        p_max = p_euler * 0.99
        
        # תיקון דינמי של גבולות החצייה במידת הצורך
        for factor in [0.99, 0.9, 0.5, 0.2, 0.05]:
            if f(p_min) * f(p_euler * factor) < 0:
                p_max = p_euler * factor
                break
        
        try:
            critical_load = bisect(f, p_min, p_max, xtol=1e-5)
        except ValueError:
            # מוצא אחרון בהחלט - החזרת הערך המשוער הליניארי
            critical_load = (sigma_allow * A) / (1 + (e * c / r**2))

    return float(critical_load)
