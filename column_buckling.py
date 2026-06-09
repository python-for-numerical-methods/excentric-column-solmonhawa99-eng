import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    מציאת העומס המקסימלי המותר P בניוטון לפי נוסחת הסקנט.
    """
    
    # פונקציית המטרה: f(P) = sigma_max(P) - sigma_allow
    def f(P):
        # חישוב האיבר שבתוך השורש: P / (E * A)
        inside_sqrt = P / (E * A)
        if inside_sqrt < 0:
            return -sigma_allow
            
        # הארגומנט של הקוסינוס ברדיאנים
        argument = (L / (2 * r)) * np.sqrt(inside_sqrt)
        
        # הגנה תיאורטית מפני אסימפטוטה של secant (כאשר cos מתקרב ל-0)
        if argument >= np.pi / 2:
            return float('inf')
            
        # נוסחת הסקנט (sec(x) = 1 / cos(x))
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(argument)))
        
        return sigma_max - sigma_allow

    # הגדרת טווח החיפוש עבור שיטת החצייה (bisect)
    p_min = 1e-5
    
    # חסם עליון תיאורטי - עומס אוילר לעמוד אידיאלי
    p_euler = (np.pi**2 * E * A * r**2) / L**2
    
    # העומס בפועל תמיד קטן מעומס אוילר בגלל האקסצנטריות
    p_max = p_euler * 0.999
    
    # הרצת שיטת החצייה
    try:
        critical_load = bisect(f, p_min, p_max, xtol=1e-5)
    except ValueError:
        # במקרה ש-p_max עדיין גבוה מדי ואינו משנה סימן, נחפש חסם נמוך יותר באופן אדפטיבי
        for factor in [0.9, 0.5, 0.1, 0.01]:
            try:
                critical_load = bisect(f, p_min, p_euler * factor, xtol=1e-5)
                break
            except ValueError:
                continue
        else:
            # הגנה אחרונה - החזרת ערך ראשוני אם החצייה נכשלה לחלוטין
            return float(p_min)

    return float(critical_load)
