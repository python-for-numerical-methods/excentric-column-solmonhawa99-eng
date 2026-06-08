import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    מחשב את העומס המקסימלי המותר (P) על עמוד אקסצנטרי לפי נוסחת הסקנט.
    """
    
    # הגדרת פונקציית העזר שחריגתה מאפס מייצגת את הפתרון [cite: 43, 45]
    def f(P):
        # חישוב הארגומנט בתוך ה-cos ברדיאנים [cite: 42, 43]
        # הערה: הנוסחה משתמשת ב-P כמשתנה תחת השורש
        term_inside_cos = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # חישוב המאמץ המקסימלי לפי נוסחת הסקנט [cite: 14]
        # sec(x) = 1 / cos(x) [cite: 42]
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(term_inside_cos)))
        
        # f(P) = sigma_max(P) - sigma_allowable 
        return sigma_max - sigma_allow

    # הגדרת טווח חיפוש נומרי בטוח עבור שיטת החצייה (Bisection) [cite: 39]
    # גבול תחתון: קרוב מאוד ל-0
    p_min = 1e-5
    
    # גבול עליון: חסם עליון תיאורטי (עומס אוילר)
    p_euler = (np.pi**2 * E * A * r**2) / L**2
    
    # כדי למנוע מצב שבו הקוסינוס מתאפס (בדיוק באוילר), ניקח 99% מעומס אוילר כגבול עליון
    p_max = p_euler * 0.99

    # הרצת שיטת החצייה לקבלת השורש המדויק [cite: 39]
    critical_load = bisect(f, p_min, p_max, xtol=1e-6)
    
    return float(critical_load)
