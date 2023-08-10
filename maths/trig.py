import sys 
sys.path.append("../")
import numpy as np

def calculate_distance_angles(row): 
    # Using dummy variables x and y to calc distance and angle attributes
    x = row['Center_dist']*.68
    y = row['y']*1.05
    row['Distance'] = np.sqrt(x**2 + y**2)
    
    c=7.32
    
    a=np.sqrt((x-7.32/2)**2 + y**2)
    b=np.sqrt((x+7.32/2)**2 + y**2)
    
    k = (c**2-a**2-b**2)/(-2*a*b)
    gamma = np.arccos(k)
    if gamma<0:
        gamma = np.pi + gamma
    
    row['Angle Radians'] = gamma
    row['Angle Degrees'] = gamma*180/np.pi
    
    return row