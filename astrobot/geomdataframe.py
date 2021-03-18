import geopandas as gpd
import pandas as pd


class GeomDataFrame(gpd.GeoDataFrame):
    """
    Ref: https://github.com/geopandas/geopandas/blob/master/geopandas/geodataframe.py
    """
    def __init__(self, *args, **kwargs):
        #key = kwargs.pop('key')
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return GeomDataFrame

    @property
    def idx(self):
        return self.index

    def add_metadata(self, metadatalst):
        self._metadata += metadatalst

    def _repr_html_(self):
        """
        Ref: https://github.com/pandas-dev/pandas/blob/master/pandas/core/frame.py#L656
        Return a html representation for a particular DataFrame.
        """
        frame = self
        droplst = ['geometry'] + [col for col in self.columns if ('geom' in col) or ('id' in col)]
        collst = [col for col in frame.columns if col not in droplst]
        frame = frame[collst]

        if frame._info_repr():
            buf = StringIO("")
            frame.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return "<pre>" + val + "</pre>"

        if pd.get_option("display.notebook_repr_html"):
            max_rows = pd.get_option("display.max_rows")
            max_cols = pd.get_option("display.max_columns")
            show_dimensions = pd.get_option("display.show_dimensions")

            return frame.to_html(
                max_rows=max_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                notebook=True,
            )
        else:
            return None

    def copy(self, deep=True, meta=False):
        """
        Make a copy of this DataFrame object.

        Borrowed from GeoPandas.GeoDataFrame.copy

        Ref: https://github.com/nilmtk/nilmtk/issues/83

        Parameters
        ----------
        deep : boolean, default True
            Make a deep copy, i.e. also copy data
        meta : boolean, default False
            Adds meta data
        Returns
        -------
        copy : ElectricDataFrame
        """

        data = self._data
        if deep:
            data = data.copy()

        return GeomDataFrame(data).__finalize__(self)