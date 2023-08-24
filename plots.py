
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
#from maths.pitch import dictCoordenadas36, draw_pitch, set_xy_coordinates
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve


def plot_distribution(df):
	#plot distributions of shots by distance and angle side by side
	fig, axes = plt.subplots(1, 2)
	distance = df.hist("Distance",bins=40,range = (0,45),ax= axes[0])
	angles = df.hist("Angle Degrees",bins=40, range = (0,100),ax=axes[1])   
	fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure1_Distributions .jpeg', format='jpeg', dpi=1200)


def plot_violin_plot(df):
	# Figure 2
	#use the seaborn library to inspect the distribution of the shots by result (goal or no goal) 
	fig, axes = plt.subplots(1, 2,figsize=(11, 5))

	#use seaborn lib for violin plot and extract necessary columns from our dataframe df
	shot_dist = sns.violinplot(x="Goal", y="Distance",
	                    data=df, inner="quart",ax= axes[0])
	shot_dist.set(xlabel="Goal? (0=no, 1=yes)",
	       ylabel="Distance (m)",
	       title="Distance of Shot from Goal vs. Result",ylim=(0, 45));

	#similar as before
	shot_ang = sns.violinplot(x="Goal", y="Angle Degrees",
	                    data=df, inner="quart",ax = axes[1])
	shot_ang.set(xlabel="Goal? (0=no, 1=yes)",
	       ylabel="Angle (Degrees)",
	       title="Shot angle vs. Result");
	fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure2_ViolinPlot.jpeg', format='jpeg', dpi=1200)


def plot_heavside(df_HS_goal,df_HS_miss):
    fig, axes = plt.subplots(figsize=(11,7))

    goals = plt.scatter(df_HS_goal['Distance'].head(100),df_HS_goal['Goal'].head(100))
    misses = plt.scatter(df_HS_miss['Distance'].head(100),df_HS_miss['Goal'].head(100),marker='x')
    plt.plot([0,12], [1, 1], 'k-', lw=2,c='green')
    plt.plot([12,12], [0, 1], 'k-', lw=2,c='green')
    plt.plot([12,40], [0, 0], 'k-', lw=2,c='green')
    plt.xlabel('Distance (m)')
    plt.ylabel('Shot Result')
    plt.title('Heaviside Function Classification for Shots')    
    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure3_Heaviside.jpeg', format='jpeg', dpi=1200)



# def plot_shots():
#     #heaviside returns 0 if y is negative and 1 otherwise
#     def heaviside(Y):
#         A = np.where(Y<=0, 0, 1)
#         return A

#     #plot contour plot over the pitch from before
#     fig, ax = plt.subplots(figsize=(11, 7))
#     draw_pitch(orientation="v",
#                aspect="half",
#                pitch_color='white',
#                line_color="black",
#                ax=ax)

#     x0 = np.linspace(0, 68, 100)
#     x1 = np.linspace(0,53 , 100)
#     x11 = np.linspace(53,0 , 100)
#     x0_grid, x1_grid = np.meshgrid(x0, x1)
#     h_grid = heaviside(144-(x0_grid-34)**2-(x1_grid-53)**2)
#     plt.contourf(x0, x11, h_grid,cmap='OrRd',alpha=.8,levels=10)
#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')
#     plt.title('Heaviside Function Classification for Shots')

#     plt.axis('off')
#     ax.set_xlim(0,68)
#     ax.set_ylim(52.5,0)
#     plt.colorbar()
#     plt.show()
#     fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure4_Shots.jpeg', format='jpeg', dpi=1200)


def plot_goal_miss(df, df_log_goal, df_log_miss):
    #same as before
    prob=np.array(df['Goal'])
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_pitch(orientation="h",
                aspect="half",
                pitch_color='white',
                line_color="black",
                ax=ax)
    goals = plt.scatter(data = df_log_goal.head(100),x='Y', y='X',alpha=.7)
    misses = plt.scatter(data = df_log_miss.head(200),x='Y', y='X',alpha=.7,marker='x')
    #classifer = Arc(xy=(10,10),width=12,height=12)
    plt.legend((goals,misses),('Goal','Miss'))
    plt.axis('off')
    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Goal_miss.jpeg', format='jpeg', dpi=1200)


def plot_logistic_regression_shots(df_log_goal, df_log_miss, lgm_dis):
    fig, axes = plt.subplots(figsize=(11,7))
    goals = plt.scatter(df_log_goal['Distance'].head(100),df_log_goal['Goal'].head(100))
    misses = plt.scatter(df_log_miss['Distance'].head(100),df_log_miss['Goal'].head(100),marker='x')
    y = np.linspace(0,40,100)
    
    plt.plot(y,1/(1+np.exp(-(lgm_dis.coef_[0][0]*y+lgm_dis.intercept_[0]))),c='Green',label='Log Model')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Shot Result')
    plt.legend()
    plt.title('Logistic Regression Model for Shots')
    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure5_LogisticRegressionShots.jpeg', format='jpeg', dpi=1200)


def plot_prob_score_base_distance_logistic(df):
    #use the seaborn library to inspect the distribution of the shots by result (goal or no goal) 
    fig, axes = plt.subplots(figsize=(11, 5))
    #first we want to create bins to calc our probability
    #pandas has a function qcut that evenly distibutes the data 
    #into n bins based on a desired column value
    df['Goal']=df['Goal'].astype(int)
    df['Distance_Bins'] = pd.qcut(df['Distance'],q=100)
    #now we want to find the mean of the Goal column(our prob density) for each bin
    #and the mean of the distance for each bin
    dist_prob = df.groupby('Distance_Bins',as_index=False)['Goal'].mean()['Goal']
    dist_mean = df.groupby('Distance_Bins',as_index=False)['Distance'].mean()['Distance']
    dist_trend = sns.scatterplot(x=dist_mean,y=dist_prob)
    dist_trend.set(xlabel="Distance (m)",
           ylabel="Probabilty of Goal",
           title="Probability of Scoring Based on Distance")
    dis = np.linspace(0,50,100)
    sns.lineplot(x = dis,y = 1/(1+np.exp((0.146*dis-.097))),color='green',legend='auto',label='Logistic Fit')
    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure6_ProbabilityScoring-Distance.jpeg', format='jpeg', dpi=1200)


def plot_prob_score_base_distance_quadratic(df):

    #use the seaborn library to inspect the distribution of the shots by result (goal or no goal) 
    fig, axes = plt.subplots(figsize=(11, 5))
    #first we want to create bins to calc our probability
    #pandas has a function qcut that evenly distibutes the data 
    #into n bins based on a desired column value
    df['Goal']=df['Goal'].astype(int)
    df['Distance_Bins'] = pd.qcut(df['Distance'],q=100)
    #now we want to find the mean of the Goal column(our prob density) for each bin
    #and the mean of the distance for each bin
    dist_prob = df.groupby('Distance_Bins',as_index=False)['Goal'].mean()['Goal']
    dist_mean = df.groupby('Distance_Bins',as_index=False)['Distance'].mean()['Distance']
    dist_trend = sns.scatterplot(x=dist_mean,y=dist_prob)
    dist_trend.set(xlabel="Distance (m)",
           ylabel="Probabilty of Goal",
           title="Probability of Scoring Based on Distance")
    dis = np.linspace(0,50,100)
    sns.lineplot(x = dis,y = 1/(1+np.exp((0.21632621*dis-0.00206089*dis**2-0.58419379))),color='green',
                 label='Log Fit with Quadratic Term')
    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure7_ProbabilityScoring-Distance.jpeg', format='jpeg', dpi=1200)


def plot_prob_score_base_angle_logistic(df,lgm_ang):
    #same for the angle
    fig, axes = plt.subplots(figsize=(11, 5))
    df['Angle_Bins'] = pd.qcut(df['Angle Degrees'],q=100)
    angle_prob = df.groupby('Angle_Bins',as_index=False)['Goal'].mean()['Goal']
    angle_mean = df.groupby('Angle_Bins',as_index=False)['Angle Degrees'].mean()['Angle Degrees']
    angle_trend = sns.scatterplot(x=angle_mean,y=angle_prob)
    angle_trend.set(xlabel="Avg. Angle of Bin",
           ylabel="Probabilty of Goal",
           title="Probability of Scoring Based on Angle")
    ang = np.linspace(0,100,100)
    sns.lineplot(x = ang,y = 1/(1+np.exp(-(lgm_ang.coef_[0][0]*ang+lgm_ang.intercept_[0]))),color='green',
                 label='Log Fit')
    
    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure8_ProbabilityScoring-Angle.jpeg', format='jpeg', dpi=1200)


def plot_prob_score_base_angle_quadratic(df, lr_ang_poly):
    fig, axes = plt.subplots(figsize=(11, 5))
    df['Angle_Bins'] = pd.qcut(df['Angle Degrees'],q=100)
    angle_prob = df.groupby('Angle_Bins',as_index=False)['Goal'].mean()['Goal']
    angle_mean = df.groupby('Angle_Bins',as_index=False)['Angle Degrees'].mean()['Angle Degrees']
    angle_trend = sns.scatterplot(x=angle_mean,y=angle_prob)
    angle_trend.set(xlabel="Avg. Angle of Bin",
           ylabel="Probabilty of Goal",
           title="Probability of Scoring Based on Angle")
    ang = np.linspace(0,100,100)
    sns.lineplot(x = ang,y = 1/(1+np.exp(-(lr_ang_poly.coef_[0][0]*ang + lr_ang_poly.coef_[0][1]*ang**2
                                           + lr_ang_poly.intercept_[0]))),color='green',
                 label='Log Fit')

    plt.show()
    fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure9_ProbabilityScoring-Angle.jpeg', format='jpeg', dpi=1200)



# def plot_xg_model(x0,x1,x_0,h_grid):
#     fig, ax = plt.subplots(figsize=(11, 7))
#     draw_pitch(orientation="vertical",
#                aspect="half",
#                pitch_color='white',
#                line_color="black",
#                ax=ax)


#     CS =plt.contourf(x_0,x1, h_grid,alpha=.85,cmap='OrRd',levels=50)


#     plt.title('xG Model')

#     #plt.axis('off')
#     ax.set_xlim(0,68)
#     ax.set_ylim(52.5,0)
#     plt.colorbar()
#     plt.show()
#     fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure10_xG-Model.jpeg', format='jpeg', dpi=1200)


# def plot_xg_model_levels(x0,x1,x_0,h_grid):
#     matplotlib.rcParams['xtick.direction'] = 'out'
#     matplotlib.rcParams['ytick.direction'] = 'out'


#     fig, ax = plt.subplots(figsize=(11, 7))
#     draw_pitch(orientation="v",
#                aspect="half",
#                pitch_color='white',
#                line_color="black",
#                ax=ax)


#     CS =plt.contour(x_0,x1, h_grid,alpha=1,cmap='OrRd',levels=7)

#     # Define a class that forces representation of float to look a certain way
#     # This remove trailing zero so '1.0' becomes '1'
#     class nf(float):
#         def __repr__(self):
#             str = '%.1f' % (self.__float__(),)
#             if str[-1] == '0':
#                 return '%.0f' % self.__float__()
#             else:
#                 return '%.1f' % self.__float__()


#     # Recast levels to new class
#     CS.levels = [nf(val*100) for val in CS.levels]

#     # Label levels with specially formatted floats
#     if plt.rcParams["text.usetex"]:
#         fmt = r'%r \%%'
#     else:
#         fmt = '%r %%'
#     plt.clabel(CS, CS.levels[1::2],inline=True, fmt=fmt, fontsize=12)

#     plt.title('xG Model')

#     #plt.axis('off')
#     ax.set_xlim(10,58)
#     ax.set_ylim(22,0)
#     plt.show()
#     fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre_gm_vitor/learn/plots/Figure11_xG-Model-Levels.jpeg', format='jpeg', dpi=1200)


# def plot_xg_model_levels_Y(x0,x1,x_0,h_grid_Y):
# 	matplotlib.rcParams['xtick.direction'] = 'out'
# 	matplotlib.rcParams['ytick.direction'] = 'out'

# 	fig, ax = plt.subplots(figsize=(11, 7))
# 	draw_pitch(orientation="vertical",
# 	           aspect="half",
# 	           pitch_color='white',
# 	           line_color="black",
# 	           ax=ax)

# 	CS =plt.contour(x_0,x1, h_grid_Y,alpha=1,cmap='OrRd',levels=10)

# 	# Define a class that forces representation of float to look a certain way
# 	# This remove trailing zero so '1.0' becomes '1'
# 	class nf(float):
# 	    def __repr__(self):
# 	        str = '%.1f' % (self.__float__(),)
# 	        if str[-1] == '0':
# 	            return '%.0f' % self.__float__()
# 	        else:
# 	            return '%.1f' % self.__float__()


# 	# Recast levels to new class
# 	CS.levels = [nf(val*100) for val in CS.levels]

# 	# Label levels with specially formatted floats
# 	if plt.rcParams["text.usetex"]:
# 	    fmt = r'%r \%%'
# 	else:
# 	    fmt = '%r %%'
# 	plt.clabel(CS, CS.levels[:3],inline=True, fmt=fmt, fontsize=10)

# 	plt.title('xG Model (with added distance to center variable)')

# 	plt.axis('off')
# 	ax.set_xlim(10,58)
# 	ax.set_ylim(22,0)
# 	plt.show()
# 	fig.savefig('/Volumes/Dados/Documents/Code/Apps/gitlab/conteudo/gm_workspace/gatomestre/learn/plots/Figure11_xG-Model-Levels.jpeg', format='jpeg', dpi=1200)



def plot_logistic_regression_model_for_shots(df_log_goal,df_log_miss, 
    lgm_dis,threshold=None,threshold_x=None):
    #plot the heaviside function on top of the responses for the seperable data above
    threshold_=threshold
    threshold_x_=threshold_x
    df_goal = df_log_goal[:100]
    df_miss = df_log_miss[:100]
    TP = df_goal[df_goal['Distance']<threshold_x_]
    TN = df_miss[df_miss['Distance']>threshold_x_]
    FP = df_goal[df_goal['Distance']>threshold_x_]
    FN = df_miss[df_miss['Distance']<threshold_x_]

    fig, axes = plt.subplots(figsize=(11,7))
    TP_scatter = plt.scatter(TP['Distance'],TP['Goal'],c='Green',label='True Positive = Correctly Pred. Goal')
    FP_scatter = plt.scatter(FP['Distance'],FP['Goal'],c='blue',label='False Positive = Incorrectly Pred. Goal',marker='x')
    TN_scatter = plt.scatter(TN['Distance'],TN['Goal'],c='orange',label='True Negative = Correctly Pred. Miss')
    FN_scatter = plt.scatter(FN['Distance'],FN['Goal'],c='red',label='False Negative = Incorrectly Pred. Miss',marker='x')

    #goals = plt.scatter(df_log_goal['Distance'].head(100),df_log_goal['Goal'].head(100))
    #misses = plt.scatter(df_log_miss['Distance'].head(100),df_log_miss['Goal'].head(100),marker='x')
    y = np.linspace(threshold_x_,40,100)
    y_2 =np.linspace(0,threshold_x_,100)

    plt.plot(y,1/(1+np.exp(-(lgm_dis.coef_[0][0]*y+lgm_dis.intercept_[0]))),c='orange',label='Missed Shot Classification')
    plt.plot(y_2,1/(1+np.exp(-(lgm_dis.coef_[0][0]*y_2+lgm_dis.intercept_[0]))),c='green',label='Goal Classification')
    plt.axhline(y=threshold_, color='cyan', linestyle='dashed',label='Threshold ='+str(threshold_))
    plt.xlabel('Distance (m)')
    plt.ylabel('Shot Result')
    plt.legend()
    plt.title('Logistic Regression Model for Shots')
    plt.show()


def plot_confusion_matrix(lgm_dis,x_test_dis,y_test_dis,threshold=[]):
    threshold=threshold
    y_pred = (lgm_dis.predict_proba(x_test_dis)[:, 1] > threshold).astype('float')
    cm_dis = confusion_matrix(y_test_dis, y_pred)
    cm_display = ConfusionMatrixDisplay(cm_dis).plot(cmap='OrRd')
    cm_display.im_.colorbar.remove()
    plt.title(f'Confusion Matrix for Threshold = {[threshold]}')
    print(cm_dis)
    return cm_dis


def plot_sensitivity_vs_specificity(cm_dis,threshold):
    #sensitivity = the ability of the model to correctly identify shots that resulted in a goal.
    sensitivity = cm_dis[1][1]/(cm_dis[1][1]+cm_dis[1][0])
    print(f'sensitivity when threshold: {threshold} = ' + str(sensitivity))
    #the ability of the model to correctly identify shots that did not result in a goal
    specificity = cm_dis[0][0]/(cm_dis[0][0]+cm_dis[0][1])
    print(f'specificity when threshold:{threshold} = '+ str(specificity) )
    print(cm_dis[0][1])
    return sensitivity, specificity


