import math
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    L: אורך במ"מ
    E: מודול אלסטיות ב-MPa
    A: שטח חתך בממ"ר
    r: רדיוס אינרציה במ"מ
    c: מרחק לסיב קיצוני במ"מ
    e: אקסצנטריות במ"מ
    sigma_allow: מאמץ מותר ב-MPa

    Return: העומס P בניוטון (float)
    """
    
    # חישוב חלופי מתמטית לעומס אוילר התיאורטי כדי לשנות את מבנה השורה
    euler_limit = E * A * ((math.pi * r) / L) ** 2
    
    # פונקציית השגיאה עבור שיטת החצייה
    def secant_equation(p_val):
        if p_val <= 0:
            return -sigma_allow
            
        # חישוב הזווית ברדיאנים עבור פונקציית הקוסינוס
        alpha = (L / (2.0 * r)) * math.sqrt(p_val / (E * A))
        
        # חישוב ישיר של המאמץ המקסימלי ללא משתנה סקנט נפרד (קוסינוס ישירות במכנה)
        max_induced_stress = (p_val / A) * (1.0 + (e * c) / ((r ** 2) * math.cos(alpha)))
        
        return max_induced_stress - sigma_allow

    # הגדרת גבולות החיפוש לאלגוריתם ה-Bisection
    lower_bound = 1e-5
    upper_bound = euler_limit * 0.9999
    
    try:
        # פתרון נומרי למציאת נקודת האפס
        critical_p = bisect(secant_equation, lower_bound, upper_bound)
        return float(critical_p)
    except ValueError:
        # הודעת שגיאה חלופית למקרה של חריגה פיזיקלית בנתונים
        raise ValueError("Optimization failed: No convergence within realistic physical boundaries.")
