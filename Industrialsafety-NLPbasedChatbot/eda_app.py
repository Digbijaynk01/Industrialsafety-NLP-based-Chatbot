# -*- coding: utf-8 -*-
# - - - - - - - - - - - Sri Pandi - - - - - - - - - - - - - -

__author__ = 'Satheesh R'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import matplotlib
import pandas as pd
import streamlit as st

# import panel as pn

matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

from pylab import *
import holoviews as hv
from holoviews import opts


@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df


def run_eda_app():
    st.subheader("EDA Analysis")
    data_set = load_data("data/Industrial_safety_df.csv")

    submenu = st.sidebar.selectbox("SubMenu", ["Descriptive", "Plots"])
    if submenu == "Descriptive":
        st.subheader("Descriptive")

        # EDA
        st.dataframe(data_set)
        with st.beta_expander("Data Types Summary"):
            st.dataframe(data_set.dtypes)

        with st.beta_expander("Descriptive Summary"):
            st.dataframe(data_set.describe())

        with st.beta_expander("Accident Level Distribution"):
            st.dataframe(data_set['Accident Level'].value_counts())

        with st.beta_expander("Potential Accident Level Distribution"):
            st.dataframe(data_set['Potential Accident Level'].value_counts())
    else:
        st.subheader("Plots")

        # Layouts
        col1, col2 = st.beta_columns([1, 1])
        with col1:

            with st.beta_expander("Dist Plot - Accident Level"):
                fig = plt.figure()
                sns.countplot(data_set['Accident Level'])
                st.pyplot(fig)

            with st.beta_expander("Dist Plot - Potential Accident Level"):
                fig = plt.figure()
                sns.countplot(data_set['Potential Accident Level'])
                st.pyplot(fig)

        with col2:
            # with st.beta_expander("Gender Distribution"):
            #     st.dataframe(data_set['Gender'].value_counts())
            #
            # with st.beta_expander("Class Distribution"):
            #     st.dataframe(data_set['class'].value_counts())
            with st.beta_expander("Dist Plot - Accident Level"):
                gen_df = data_set['Accident Level'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Accident Level', 'Counts']
                p01 = px.pie(gen_df, names='Accident Level', values='Counts')
                st.plotly_chart(p01, use_container_width=True)

            with st.beta_expander("Dist Plot - Potential Accident Level"):
                gen_df = data_set['Potential Accident Level'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Potential Accident Level', 'Counts']
                p02 = px.pie(gen_df, names='Potential Accident Level', values='Counts')
                st.plotly_chart(p02, use_container_width=True)

        with st.beta_expander("Accident Levels by Gender"):
            # chart_data = pd.DataFrame(data_set, columns=['Industry Sector', 'Country'])
            # 
            # indsec_cntry_table = pd.crosstab(index=data_set['Industry Sector'], columns=data_set['Country'])
            # indsec_cntry_table.plot(kind='bar', figsize=(8, 8), stacked=True)
            # st.area_chart(chart_data)

            hv.extension("bokeh", "matplotlib")

            f = lambda x: np.round(x / x.sum() * 100)

            ac_gen = data_set.groupby(['Gender', 'Accident Level'])['Accident Level'].count().unstack().apply(f,
                                                                                                              axis=1)
            ac = hv.Bars(pd.melt(ac_gen.reset_index(), ['Gender']), ['Gender', 'Accident Level'], 'value').opts(
                opts.Bars(title="Accident Level by Gender Count"))

            pot_ac_gen = data_set.groupby(['Gender', 'Potential Accident Level'])[
                'Potential Accident Level'].count().unstack().apply(f, axis=1)
            pot_ac = hv.Bars(pd.melt(pot_ac_gen.reset_index(), ['Gender']), ['Gender', 'Potential Accident Level'],
                             'value').opts(opts.Bars(title="Potential Accident Level by Gender Count"))

            plot = (ac + pot_ac).opts(
                opts.Bars(width=800, height=400, tools=['hover'], show_grid=True, xrotation=0, ylabel="Percentage",
                          yformatter='%d%%'))

            # bokeh_obj = hv.render(plot, backend="bokeh")
            bokeh_obj = hv.render(plot, backend="matplotlib")
            st.plotly_chart(bokeh_obj)

        with st.beta_expander("Employee type by Gender / Industry Sector by Gender"):

            # Layouts
            col1, col2 = st.beta_columns([1, 1])
            with col1:
                hv.extension("bokeh", "matplotlib")
                f = lambda x: np.round(x / x.sum() * 100)
                em_gen = data_set.groupby(['Gender', 'Employee type'])['Employee type'].count().unstack().apply(f,
                                                                                                                axis=1)

                plot = hv.Bars(pd.melt(em_gen.reset_index(), ['Gender']), ['Gender', 'Employee type'],
                               'value').opts(opts.Bars(title="Employee type by Gender Count", width=800, height=300,
                                                       tools=['hover'], show_grid=True, xrotation=0,
                                                       ylabel="Percentage",
                                                       yformatter='%d%%'))

                bokeh_obj_01 = hv.render(plot, backend="matplotlib")
                st.plotly_chart(bokeh_obj_01)
            with col2:
                hv.extension("bokeh", "matplotlib")
                f = lambda x: np.round(x / x.sum() * 100)
                em_gen = data_set.groupby(['Gender', 'Industry Sector'])['Industry Sector'].count().unstack().apply(f,
                                                                                                                    axis=1)

                plot = hv.Bars(pd.melt(em_gen.reset_index(), ['Gender']), ['Gender', 'Industry Sector'],
                               'value').opts(opts.Bars(title="Industry Sector by Gender Count", width=800, height=300,
                                                       tools=['hover'], show_grid=True, xrotation=0,
                                                       ylabel="Percentage",
                                                       yformatter='%d%%'))

                bokeh_obj_02 = hv.render(plot, backend="matplotlib")
                st.plotly_chart(bokeh_obj_02)
