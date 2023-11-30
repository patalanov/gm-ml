import sys 
sys.path.append("../")
import numpy as np

def calculate_distance_angles(row,metric): 
    # Using dummy variables x and y to calc distance and angle attributes
    if metric == 'xG':
        x = row['Center_dist']*.68
        y = row['y']*1.05
        row['Distance'] = np.sqrt(x**2 + y**2)

    if metric == 'xA':
        x = row['Center_dist_quadrante_event']*.68
        y = row['y_quadrante_event']*1.05
        row['Distance_quadrante_event'] = np.sqrt(x**2 + y**2)

    # if metric == 'xPre':
    #     x = row['Center_dist PreAssist']*.68
    #     y = row['y_preassist']*1.05
    #     row['Distance PreAssist'] = np.sqrt(x**2 + y**2)

    if metric == 'xGOT':
        x = row['Center_dist Ot']*.68
        y = row['y_ot']*1.05
        row['Distance Ot'] = np.sqrt(x**2 + y**2)

    #row['Distance'] = np.sqrt(x**2 + y**2)  
    c=7.32
    
    a=np.sqrt((x-7.32/2)**2 + y**2)
    b=np.sqrt((x+7.32/2)**2 + y**2)
    
    k = (c**2-a**2-b**2)/(-2*a*b)
    gamma = np.arccos(k)
    if gamma<0:
        gamma = np.pi + gamma
    

    if metric == 'xG':
        row['Angle Radians'] = gamma
        row['Angle Degrees'] = gamma*180/np.pi
    if metric == 'xA':
        row['angle_radians_event'] = gamma
        row['angle_degrees_event'] = gamma*180/np.pi
    # if metric == 'xPre':
    #     row['Angle Radians PreAssist'] = gamma
    #     row['Angle Degrees PreAssist'] = gamma*180/np.pi
    if metric == 'xGOT':
        row['Angle Radians Ot'] = gamma
        row['Angle Degrees Ot'] = gamma*180/np.pi
    
    return row



def calculate_angles(row, metric):
    """
    Calculates the horizontal and vertical angles of a shot based on shot coordinates.
    
    Parameters:
    row (Series): A row from a DataFrame containing the shot coordinates in meters.
    metric (str): The metric name used to store the angle results.
    
    Returns:
    Series: The input row augmented with angle calculations.
    
    The angles are calculated based on the positions given in meters.
    Horizontal angle is calculated with respect to the center of the goal width.
    The vertical angle is assumed to start at 0 degrees from the ground, becoming positive as it rises.
    """
    
    goal_width_m = 7.32  # Official goal width in meters

    # Coordinates in meters are already provided in the dataframe
    if metric == 'xG':
        x_m = row['field_x_metros']
        y_m = row['field_y_metros']  # This is the position on the field, not the height above the ground
    elif metric == 'xGOT':
        x_m = row['field_x_metros_on_target']
        y_m = row['field_y_metros_on_target']  # This is the position on the field, not the height above the ground
    elif metric == 'xA':
        x_m = row['field_x_metros_event']
        y_m = row['field_y_metros_event'] # the event can be either a shot, an assist or a preassist
    
    # Horizontal angle calculation with respect to the goal center
    a = np.sqrt((x_m - goal_width_m / 2) ** 2 + y_m ** 2)
    b = np.sqrt((x_m + goal_width_m / 2) ** 2 + y_m ** 2)
    cos_gamma = (goal_width_m ** 2 - a ** 2 - b ** 2) / (-2 * a * b)
    gamma = np.arccos(np.clip(cos_gamma, -1, 1))  # Clip for numerical stability

    # The vertical angle calculation is simplified since all shots are assumed to be taken at 0 degrees
    # 'goal_y_px' is the height at which the ball crosses the goal line, so it is the vertical component
    vertical_angle_radians = 0  # Starts at 0 degrees from the ground

    # If 'goal_y_px' is provided, calculate the vertical angle at which the ball crosses the goal line
    if metric == 'xG':
        if 'goal_y_px' in row and 'goal_x_px' in row:
            goal_x_px = row['goal_x_px']
            goal_y_px = row['goal_y_px']
            vertical_angle_radians = np.arctan2(goal_y_px, goal_x_px)  # Use arctan2 to calculate the angle
    elif metric == 'xGOT':
        if 'goal_y_px_on_target' in row and 'goal_x_px_on_target' in row:
            goal_x_px = row['goal_x_px_on_target']
            goal_y_px = row['goal_y_px_on_target']
            vertical_angle_radians = np.arctan2(goal_y_px, goal_x_px)  # Use arctan2 to calculate the angle
    elif metric == 'xA':
        if 'goal_y_px_event' in row and 'goal_x_px_event' in row:
            goal_x_px = row['goal_x_px_event']
            goal_y_px = row['goal_y_px_event']
            vertical_angle_radians = np.arctan2(goal_y_px, goal_x_px)  # Use arctan2 to calculate the angle

    # Storing results
    row[metric + '_angle_radians'] = gamma
    row[metric + '_angle_degrees'] = np.degrees(gamma)
    row[metric + '_vertical_angle_radians'] = vertical_angle_radians
    row[metric + '_vertical_angle_degrees'] = np.degrees(vertical_angle_radians)

    return row





