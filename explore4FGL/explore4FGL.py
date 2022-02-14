import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns

import astropy.coordinates as coord
import astropy.units as u
import pandas as pd
from numpy import float64
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import metrics

import logging

import pandas

logging.basicConfig(level=logging.INFO)

output_path = f"{os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))}/output"


class Source4FGLData:
    """
    A class representing the data fittings from the "Fermi Gamma-ray Space Telescope" in the fourth catalog (4FGL).
    With this it is possible to make a spatial representation of the sources, but also to relate fields of the dataframe
    in an arbitrary way. An attempt to classify the sources by means of a machine learning algorithm is also implemented

    Parameters
    ----------
    pandas_df : pandas DataFrame
        Data to initialize Source4FGLData.
    """

    def __init__(self, pandas_df: pandas.DataFrame):
        self._df = pandas_df.apply(lambda x: x.str.strip().str.upper() if x.dtype == "object" else x)
        logging.info(f'All string columns of the dataframe have been rewritten in uppercase')

        # Create output dir if it doesn't exist
        if not os.path.isdir(output_path):
            warnings.warn(f"The output folder doesn't exists")
            os.makedirs(output_path)
            logging.info(f"The output path has been created in {output_path}")

    @property
    def df(self):
        """
        With this workaround we are sure not to modify the dataframe parameter of the class in any way
        :return: pandas DataFrame
            A copy of the source dataframe
        """
        return self._df.copy()

    @classmethod
    def from_4fgl_file(cls, fits_path):
        """
        Get a Source4FGLData object from a 4FGL file fits
        :param fits_path: file fits path
        :type fits_path: str
        :return: a Source4FGLData object containing only one-dimensional columns
        :rtype: Source4FGLData
        """
        with fits.open(fits_path) as hdu_list:
            astropy_table = Table(hdu_list[1].data)
        # Filter only 1D columns
        columns_1d = [col1D for col1D in astropy_table.colnames if len(astropy_table[col1D].shape) <= 1]
        logging.info(f'Used only 1D columns')
        return cls(astropy_table[columns_1d].to_pandas())

    def _clean_df(self, columns):
        """
        (internal method) Remove rows from the dataframe that have the value NaN in certain columns
        :param columns: list(str)
            list of columns where to look for NaN values
        :return: pandas DataFrame
        """
        logging.info(f'Remove rows with NaN value on fields {columns}')
        try:
            not_none_columns = [col for col in columns if col is not None]
            df_cleaned = self.df.dropna(subset=not_none_columns)
        except KeyError as e:
            logging.error(f"column {e} not present in {self.df.columns}")
            sys.exit(1)
        return df_cleaned

    def _check_numeric_type(self, col):
        """
        (internal method) Check if specified column is a numeric type
        :param col: str
            column of pandas DataFrame
        :return: None
            exit error 2 if column type isn't numeric
        """
        if not is_numeric_dtype(self.df[col]):
            logging.error(f"column {col} is not numeric type, select one of the following fields:\n"
                          f"{self.df.select_dtypes(include=np.number).columns}")
            sys.exit(2)

    @staticmethod
    def save_or_show_plot(savefig=False, title='Title'):
        """
        Shows or saves the plot in the output folder
        :param savefig: boolean
            if True save the figure in the output directory, else only shows it
        :param title: str
            title of the plot (default='Title')
        """
        if savefig:
            plt.savefig(f"{output_path}/{title}.png")
            logging.info(f'Image saved in {output_path}/{title}.png')
        else:
            logging.info(f'The image {title} is shown')
            plt.show()

    def get_hist(self, colname, title='Histogram', savefig=False, xlog=False, ylog=False, **kwargs):
        """
        Constructs the histogram of a dataframe field
        :param colname: str
            The name of the column to plot. The column must be numeric
        :param title: str
            The title of the histogram
        :param savefig: boolean
        :param xlog: boolean
            if True, set the abscissa axis to logarithmic scale
        :param ylog: boolean
            if True, set the ordinate axis to logarithmic scale
        :param kwargs: dict
            other parameters are passed to matplotlib.pyplot.hist module
        """
        df_cleaned = self._clean_df([colname])

        data = df_cleaned[colname]

        logging.info('Preparing the histogram')

        plt.figure()
        plt.hist(data, **kwargs)
        plt.title(title)
        plt.xlabel(colname)
        plt.ylabel('Counts')
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')

        logging.info('Histogram is ready')
        self.save_or_show_plot(savefig=savefig, title=title)

    def get_plot(self, col_x, col_y, title=None, savefig=False, xlog=False, ylog=False, **kwargs):
        """
        Get the plot of the data by relating two numeric fields of the dataframe
        :param col_x: str
            The name variable of the abscissa. The column must be numeric
        :param col_y: str
            The name variable of the ordinate. The column must be numeric
        :param title: str
            The title of the histogram
        :param savefig: boolean
        :param xlog: boolean
            if True, set the abscissa axis to logarithmic scale
        :param ylog: boolean
            if True, set the ordinate axis to logarithmic scale
        :param kwargs: dict
            other parameters are passed to matplotlib.pyplot.hist module
        """
        df_cleaned = self._clean_df([col_x, col_y])

        if title is None:
            title = f'{col_x} vs {col_y}'

        self._check_numeric_type(col_x)
        self._check_numeric_type(col_y)
        x = df_cleaned[col_x]
        y = df_cleaned[col_y]

        logging.info(f'Preparing the "{col_x} vs {col_y}" plot')
        plt.figure()
        plt.scatter(x, y, marker='.', **kwargs)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.xlabel(col_x)
        plt.ylabel(col_y)

        logging.info('Plot is ready')
        self.save_or_show_plot(savefig=savefig, title=title)

    def galactic_visualization_plot(self, coord_type='equatorial',
                                    title='Galactic Map',
                                    savefig=True, color=None, **kwargs):
        """
        Get the galactic map plot with equatorial or galactic coordinates for given sources
        :param coord_type: str
            type of the given coordinates. Possible values are "equatorial" and "galactic" (default="equatorial")
        :param title: str
            title of the histogram shown in the plot (string type)
        :param savefig: boolean
        :param color: str, int or None
            the name of the column to color the points. If columns is string type, the axes will be drawn with
            'seaborn.scatterplot' module, if the column is numeric type 'matplotlib.pyplot.scatter' will take its
            place
        :param kwargs: dict
            other parameters are passed to matplotlib.pyplot.hist module
        """
        try:
            assert (coord_type in ['equatorial', 'galactic'])
        except Exception:
            raise ValueError('coord_type must be "equatorial" or "galactic"')

        color_label = color

        lon_label, lat_label = ["RAJ2000", "DEJ2000"] if coord_type == 'equatorial' else ["GLON", "GLAT"]
        df_cleaned = self._clean_df([lon_label, lat_label, color])

        lat = df_cleaned[lat_label]
        # Shifted values of -180° (rather that between 0° and +360° they will be between -180° and +180°)
        lon = df_cleaned[lon_label] - 180

        # Convert deg values to radian
        lat = coord.Angle(lat * u.deg).radian
        lon = coord.Angle(lon * u.deg).radian

        logging.info(f'Preparing the Galactic plot')
        fig, ax = plt.subplots(1, 1)
        ax = plt.axes(projection='aitoff')
        ax.grid(visible=True)

        if color is not None:
            col = df_cleaned[color_label]
            coord_df = pd.DataFrame({lon_label: lon, lat_label: lat, color_label: col})
            if is_string_dtype(df_cleaned[color]):
                sns.scatterplot(x=lon_label, y=lat_label, hue=color_label, data=coord_df, **kwargs)
                ax.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, -0.05))
            elif is_numeric_dtype(df_cleaned[color_label]):
                scat = ax.scatter(lon, lat, c=col.tolist(), **kwargs)
                ax.set_xlabel(lon_label)
                ax.set_ylabel(lat_label)
                cbar = fig.colorbar(scat)
                cbar.set_label(color_label)
            fig.set_size_inches(10, 7.5)
        else:
            ax.scatter(lon, lat, **kwargs)
        ax.set_title(title)

        logging.info('Galactic Map is ready')
        self.save_or_show_plot(savefig=savefig, title=title)

    def predict_classification(self, threshold=10):
        """
        Returns a prediction on the type of source for the unclassified sources of the catalog using the DT (Decision
        Tree) algorithm having as "target" the classification (field "CLASS1") and as "feature" all the other columns of
        integer type.
        It is possible to decide which types of sources not to consider based on their populousness through the
        threshold parameter
        :param threshold: int
            filter on the minimum number of records belonging to a type of source.
        """
        df = self.df
        # Replace white space values in CLASS1 with new_val
        new_val = "NO INFO"
        df["CLASS1"] = df["CLASS1"].replace("", new_val)

        # Remaps every possible value (String) to an integer value
        possible_values = df["CLASS1"].unique()
        possible_values.sort()
        remap = {}
        reverse_remap = {}
        for index, val in enumerate(possible_values):
            remap[val] = index
            reverse_remap[index] = val
        df['CLASS1'] = df['CLASS1'].map(remap)

        # Filter out sparsely populated sources
        grouped = df.groupby("CLASS1").count()
        possible_values_filtered = grouped.loc[grouped["Source_Name"] > threshold].index
        logging.info(f'The following sources were discarded for the DT:\n'
                     f'{[reverse_remap[ind] for ind in grouped.loc[grouped["Source_Name"] <= threshold].index]}')

        # Construction of the query to extract the records belonging to the remaining sources
        list_string = "("
        for index, val in enumerate(possible_values_filtered):
            if index < len(possible_values_filtered) - 1:
                list_string += f"{val}, "
            else:
                list_string += f"{val})"
        df_filtered = df.query(f"CLASS1 in {list_string}")

        x_with_na = df_filtered[df_filtered['CLASS1'] != remap[new_val]].select_dtypes(exclude='object')  # Features
        x = x_with_na.fillna(x_with_na.mean())  # Replace missing data with the mean of the column

        # I divide the data in such a way that for each type of source 30% of the records go to rehearsals and the
        # remaining 70% to training
        x_train = pd.DataFrame([])
        x_test = pd.DataFrame([])
        y_train = pd.Series(dtype=float64)
        y_test = pd.Series(dtype=float)
        for source_type in possible_values_filtered:
            if source_type != remap[new_val]:
                sub_x = x.loc[x.CLASS1 == source_type]
                sub_y = sub_x.CLASS1
                sub_x_train, sub_x_test, sub_y_train, sub_y_test = \
                    train_test_split(sub_x, sub_y, test_size=0.3)  # 70% training and 30% test
                x_train = pd.concat([x_train, sub_x_train])
                x_test = pd.concat([x_test, sub_x_test])
                y_train = pd.concat([y_train, sub_y_train])
                y_test = pd.concat([y_test, sub_y_test])

        # Create Decision Tree classifier object
        clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=10, min_samples_leaf=4,
                                     max_depth=4)  # Prune operation to avoid overfitting

        logging.info('Generated the Decision Tree Classifier')

        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Generate accuracy curves
        train_sizes, train_scores, validation_scores = learning_curve(estimator=clf,
                                                                      n_jobs=-1,
                                                                      X=x_train,
                                                                      y=y_train,
                                                                      cv=5,
                                                                      shuffle=True,
                                                                      train_sizes=np.linspace(1, 1200, dtype=int),
                                                                      )
        train_scores_mean = train_scores.mean(axis=1)
        validation_scores_mean = validation_scores.mean(axis=1)

        # Plot accuracy curves
        logging.info('Preparing the learning curves plot')
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, label='Training accuracy')
        plt.plot(train_sizes, validation_scores_mean, label='Validation accuracy')
        plt.ylim(0, 1)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title('Decision Tree curve', fontsize=18, y=1.03)
        plt.legend()

        logging.info('The learning curves plot is ready')
        self.save_or_show_plot(True, "DT_Accuracy")

        x_unassociated_with_na = df[df['CLASS1'] == remap[new_val]].select_dtypes(exclude='object')
        x_unassociated = x_unassociated_with_na.fillna(x.mean())
        y_predict_unassociated = clf.predict(x_unassociated)
        y_predict_unassociated_reverse = [reverse_remap[val] for val in y_predict_unassociated]
        [print(f"{x} -> {y_predict_unassociated_reverse.count(x)}") for x in set(y_predict_unassociated_reverse)]
        print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
