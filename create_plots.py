import shutil
import os
import pandas as pd
from post_processing_utils.plot_builder import *
from pylatex import Document, Section, Figure, SubFigure, NoEscape
from pylatex.base_classes import Environment
from pylatex.package import Package

class subfigure(Environment):
    """A class to wrap LaTeX's alltt environment."""
    packages = [Package('subcaption')]
    escape = False
    content_separator = "\n"
    _repr_attributes_mapping = {
        'position': 'options',
        'width': 'arguments',
    }

    def __init__(self, position=NoEscape(r'H'),width=NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.
        """

        super().__init__(options=position,arguments=width, **kwargs)

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 90)
pd.set_option('display.expand_frame_repr', False)

dim_theta_phi = {
    1:{'d_x':1,'d_y':1,'d_z':1,'theta':2.0,'phi':2.0},
    3:{'d_x':3,'d_y':3,'d_z':3,'theta':4.0,'phi':2.0},
    15:{'d_x':3,'d_y':3,'d_z':15,'theta':8.0,'phi':2.0},
    50:{'d_x':3,'d_y':3,'d_z':15,'theta':16.0,'phi':2.0},
}

def plot_1_true_weights():

    all_path = 'do_null_100/'
    small_path = 'base_jobs_kc_layers=3_width=32'
    df = pd.read_csv('base_jobs_kc.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['d_Z','n'])
    data_paths=[]
    suffix_paths=[]
    for index,row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'],required_n=row['n'],estimator=row['nce_style'],br=500))
        data_paths.append(build_path(dx=dim_theta_phi[dz]['d_x'],
                                     dy=dim_theta_phi[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi[dz]['theta'],
                                     phi=dim_theta_phi[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0,0.0],
                                     yz=[0.5,0.0])
                          )
    DIRNAME = 'plot_1_real'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}.jpg')

    doc = Document(default_filepath='subfigures')
    with doc.create(Figure(position='h!')) as plot:
        for i, j, p in zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths):
            with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                path_str = '{'+p+'}'
                string_append = f'\includegraphics[width=0.33\linewidth]{path_str}%\n'
                doc.append(string_append)
                pass
    doc.generate_pdf(clean_tex=False)
if __name__ == '__main__':
    plot_1_true_weights()


