# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:09:47 2024

@author: Diego de Sousa
"""


def select_columns(df, column_positions, bool_column):
    """
    This functions make it easier to create a new df
    using the columns positions also drop na rows and
    convert boolean variable to binary integer
    """
    the_columns = [df.columns[i] for i in column_positions]
    new_df = df[the_columns].copy()
    new_df = new_df.dropna()
    new_df[bool_column] = new_df[bool_column].astype(int)
    
    return new_df


import matplotlib.pyplot as plt


def scatter_plot(data, x_column, y_column, x_label):
    """
    A function to facilitate the creation of a scatter 
    plot.
    """
    plt.scatter(x=data[x_column], y=data[y_column], color='green')
    plt.ylabel('fraud(1)/normal(0)')
    plt.xlabel(x_label)
    plt.show()




def logistic_curve(x_var, y_var, data, prob):

    plt.scatter(x_var, y_var, color='green', s=100)
    
    plt.plot(data, prob, color='black', linewidth=3)
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    
    plt.ylabel('probability of fraud', fontsize=30)
    plt.xlabel('amount', fontsize=30)
    plt.xlim(-1, 25)
    plt.tight_layout()
    
    plt.show()

