# import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


class UCB:

    def __init__(self):
        """
        explanation: Instantiates instance of UCB
        input: None
        output: None

        attributes:
            data - pd.DataFrame - raw data
            rates - pd.DataFrame - acceptance rates for males and females across department
            male - dict - holds entries for mean and standard deviation for rates DataFrame
            female - dict - holds entries for mean and standard deviation for rates DataFrame
        """
        # load data
        self.data = pd.read_csv("UC_Berkeley.csv")
        # collect data and graphs
        self.rates = self.Collect()
        # get stats
        self.male,self.female = self.Get_Stats()
        return

    #===========================================================================
    # public
    #===========================================================================

    def Total_Admission_Rates(self,graph_kwargs={},**kwargs):
        """
        explanation:
            Finds the overall acceptance rate for males and females and produces a bar graph to show the results.
        input:
            graph_kwargs: dict - optional kwargs to feed into matplotlib.axes.Axes.bar
            kwargs:
                show: bool - controls whether graph is shown - default = True
                save: bool - controls whether figure is saved - default = False
                savename: str - filename figure will be saved under - default = 'UCB_Totals.pdf'
                title: str - title of the figure - default = 'UC Berkeley Admission Rates'
        output: dict - keys: 'female_rate' & 'male_rate'
        """
        # kwargs
        show = True
        save = False
        savename = "UCB_Totals.pdf"
        title = "UC Berkeley Admission Rates"
        if 'show' in kwargs: show = kwargs['show']
        if 'save' in kwargs: save = kwargs['save']
        if 'savename' in kwargs: savename = kwargs['savename']
        if 'title' in kwargs: title = kwargs['title']

        # aggregrate data by Gender and Admission
        data = self.data.copy().groupby(by=["Gender","Admit"]).sum()

        # make plot with axes
        fig,ax = plt.subplots()
        fig.suptitle(title)
        totals = self.__graph_bar_axis(data, ax, None, graph_kwargs, **kwargs)


        if show: plt.show()
        if save:
            fig.savefig(savename)
            print("\nSaved to %s.\n" % savename)

        plt.close(fig)
        return totals

    def Admission_Rates_By_Dept(self,graph_kwargs={},**kwargs):
        """
        explanation:
            Finds the acceptance rate for males and females across departments and produces a bar graph to show the results.
        input:
            graph_kwargs: dict - optional kwargs to feed into matplotlib.axes.Axes.bar
            kwargs:
                show: bool - controls whether graph is shown - default = True
                save: bool - controls whether figure is saved - default = False
                savename: str - filename figure will be saved under - default = "UCB_Depts.pdf"
                title: str - title of the figure - default = "UC Berkeley Admission Rates by Department"
                nrows: int - number of rows for the subplots - default = 2
                ncols: int - number of columns for the subplots - default = 3
        output: dict - keys: 'female_rate' & 'male_rate'
        """

        # kwargs
        show = True
        save = False
        savename = "UCB_Depts.pdf"
        title = "UC Berkeley Admission Rates by Department"
        nrows = 2
        ncols = 3
        if 'show' in kwargs: show = kwargs['show']
        if 'save' in kwargs: save = kwargs['save']
        if 'savename' in kwargs: savename = kwargs['savename']
        if 'title' in kwargs: title = kwargs['title']
        if 'nrows' in kwargs: nrows = kwargs['nrows']
        if 'ncols' in kwargs: ncols = kwargs['ncols']

        # make array of department names
        depts = np.array(["A", "B", "C", "D", "E", "F"], dtype='O').reshape([nrows,ncols])
        rates = np.empty_like(depts, dtype='O')

        # grab whole data set
        data = self.data.copy()

        # make plot with axes
        fig,ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        fig.suptitle(title)
        plt.subplots_adjust(hspace=.3)

        # go through each department and check admission rates
        for i in range(nrows):
            for j in range(ncols):
                # get dept name for iteration
                dept_name = depts[i,j]
                # filter by dept, then aggregrate data by Gender and Admission
                dept_data = data[data.Dept == dept_name].groupby(by=["Gender","Admit"]).sum()

                # graph department admission rates
                rates[i,j] = self.__graph_bar_axis(dept_data, ax[i,j], "Dept %s" % dept_name, graph_kwargs, **kwargs)

        if show: plt.show()
        if save:
            fig.savefig(savename)
            print("\nSaved to %s.\n" % savename)
            plt.close(fig)

        return rates

    def Collect(self):
        """
        explanation:
            collects the overall rates (and does nothing with them right now). Collects the cross department admission rates and packages them into a pandas DataFrame
        input: None
        output: pd.DataFrame - female and male acceptance rates across departments
        """
        total_rates = self.Total_Admission_Rates(show=False,save=True)
        dept_rates = self.Admission_Rates_By_Dept(show=False,save=True).flatten()
        female_rates = np.zeros(6)
        male_rates = np.zeros(6)
        for i,d in enumerate(dept_rates):
            # pdb.set_trace()
            female_rates[i] = d["female_rate"]
            male_rates[i] = d["male_rate"]

        ratesDF = pd.DataFrame( dict(female=female_rates, male=male_rates) )
        return ratesDF

    def Get_Stats(self):
        """
        explanation:
        input: None
        output: tuple of dictionaries with keys: 'mean' & 'std' for both females and males
        """
        male = dict(mean="{:0.3f}".format(self.rates.male.mean()), std="{:0.3f}".format(self.rates.male.std()))
        female = dict(mean="{:0.3f}".format(self.rates.female.mean()), std="{:0.3f}".format(self.rates.female.std()))
        return male,female

    #===========================================================================
    # private
    #===========================================================================

    def __graph_bar_axis(self,data,ax,ax_title,graph_kwargs,**kwargs):
        """
        explanation:
            Used finding the admission rates for a given DataFrame for both females and males and contructs a graph with the information.
        input:
            data: pd.DataFrame - either put the raw Data in here, or filter or group the raw data and feed it into this private method.
            ax: matplotlib.Axes.ax - the figure and axis are created outside this method and the axis is fed into this private method either whole or in part.
            ax_title: str - the title of the matplotlib.Axes.ax instance
            graph_kwargs: dict - kwargs to feed into the matplotlib.Axes.ax.plot method.
            kwargs:
                fontsize: int - what fontsize the annotations appear on the figure
        output: dict - keys: 'female_rate' & 'male_rate'
        """
        # kwargs
        fontsize = 10
        if 'fontsize' in kwargs: fontsize = kwargs['fontsize']

        # separate data by male and female
        Female = data.loc["Female"]
        Male = data.loc["Male"]
        # find acceptance rates
        female_rate = (Female.loc['Admitted']/Female.sum())[0]
        male_rate = (Male.loc['Admitted']/Male.sum())[0]

        # plot axis
        f,m = ax.bar(["Female","Male"], [female_rate,male_rate], **graph_kwargs)
        f.set_facecolor('r')
        m.set_facecolor('b')
        ax.set_xlabel("Gender")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title(ax_title)
        ax.set_ylim(0,1)
        ax.annotate("{:0.3f}".format(female_rate), xy=(-.2,female_rate*1.1), color='r', fontsize=fontsize)
        ax.annotate("{:0.3f}".format(male_rate), xy=(.8,male_rate*1.1), color='b', fontsize=fontsize)

        return dict(female_rate=female_rate, male_rate=male_rate)
