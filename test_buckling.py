import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    מחשב את העומס המקסימלי המותר (P) על עמוד אקסצנטרי לפי נוסחת הסקנט.
    
    פרמטרים (לפי הגדרות המטלה):
    L (float): אורך העמוד במ"מ
    E (float): מודול האלסטיות ב-MPa
    A (float): שטח חתך בממ"ר
    r (float): רדיוס אינרציה בס"מ (מ"ס) -> יומר למ"מ
    c (float): מרחק לציר הנייטרלי לסיב הקיצוני בס"מ (מ"ס) -> יומר למ"מ
    e (float): אקסצנטריות בס"מ (מ"ס) -> יומר למ"מ
    sigma_allow (float): מאמץ מותר ב-MPa
    
    החזרה:
    float: העומס הקריטי P בניוטון
    """
    
    # המרת יחידות מס"מ (מ"ס) למילימטרים (מ"מ) עבור הנוסחה
    r_mm = r * 10.0
    c_mm = c * 10.0
    e_mm = e * 10.0
    
    # הגדרת פונקציית העזר: f(P) = sigma_max(P) - sigma_allow
    def f(P):
        # חישוב הארגומנט שבתוך ה-cos (ברדיאנים)
        term_inside_cos = (L / (2 * r_mm)) * np.sqrt(P / (E * A))
        
        # הגנה מפני חלוקה באפס או ערכים לא חוקיים ב-cos
        if term_inside_cos >= np.pi / 2:
            return float('inf')
            
        # נוסחת הסקנט (sec(x) = 1 / cos(x))
        sigma_max = (P / A) * (1 + (e_mm * c_mm / r_mm**2) * (1 / np.cos(term_inside_cos)))
        
        return sigma_max - sigma_allow

    # הגדרת טווח החיפוש (brackets) עבור שיטת החצייה
    p_min = 1e-5 
    
    # חסם עליון מוחלט: עומס אוילר התיאורטי (במילימטרים)
    p_euler = (np.pi**2 * E * A * r_mm**2) / L**2
    
    # העומס האמיתי תמיד יהיה קטן מעומס אוילר בגלל האקסצנטריות.
    # נשתמש ב-99% מעומס אוילר כנקודת קצה עליונה בטוחה לחיפוש.
    p_max = p_euler * 0.99
    
    # הרצת שיטת החצייה למציאת השורש
    critical_load = bisect(f, p_min, p_max, xtol=1e-5)
    
    return float(critical_load)
