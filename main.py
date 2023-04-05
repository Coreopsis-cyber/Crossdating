import helper, statistical_method, graphical
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import PySimpleGUI as sg
import matplotlib.pyplot as plt


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


if __name__ == "__main__":

    def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
        if canvas.children:
            for child in canvas.winfo_children():
                child.destroy()
        if canvas_toolbar.children:
            for child in canvas_toolbar.winfo_children():
                child.destroy()
        figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
        figure_canvas_agg.draw()
        toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
        toolbar.update()
        figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

    def make_win1():
        sg.theme("DarkGreen")
        layout = [
            [sg.Text("Welcome to this basic crossdating application, you will be required to input \nseveral decision "
                     "about your "
                     "data in order this program to complete."), sg.Text('      ', k='-OUTPUT-')],
            [sg.Button('Upload CSV file'), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, size=(800, 300), finalize=True)


    def make_win2():
        layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],
                  [sg.Button("Import"), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, size=(800, 300), finalize=True)


    def make_win3():
        layout = [[sg.Text("Below are two methods for crossdating your data, a machine learning method and a "
                           "statstical method. Both will produce graphics at the end ")], [sg.Button("Statistical "
                                                                                                     "Method"),
                                                                                           sg.Button(
                                                                                               "Machine Learning Method")]]
        return sg.Window('Crossdating App', layout, size=(800, 300), finalize=True)


    def make_win4():
        labels = ('Segment size:', 'Number of standard deviations away from the mean which is considered an outlier:',
                  'Number of consecutive outliers:', 'Number of start dates the program outputs:')
        keys = ('-SEGMENT-', '-STANDDEV-', '-OUTLIERS-', '-STARTDATE-')
        default = ['10', '3', '8', '1']
        size = max(map(len, labels))
        layout = [[sg.Text("Statistical Method based on pairwise lead lag analysis.\n You will be asked to to make "
                           "decisions that will effect the ability of the program to crossdate the sample.\n At each "
                           "step a default will be suggested.\n")],
                  [
                      [sg.Text(labels[i], size=size), sg.Input(default[i], key=keys[i].split()[0])]
                      for i in range(len(labels))] + [[sg.Push(), sg.Button("Submit")]],
                  ]
        return sg.Window('Crossdating App', layout, size=(850, 300), finalize=True)


    def make_win5():
        layout = [[sg.Text("Computation is occuring please wait.")],
                  [sg.ProgressBar(100, orientation='h', expand_x=True, key='-PBAR-', size=(20, 20))],
                  [sg.Text('', key='-OUT-', enable_events=True, justification='center', expand_x=True)],
                  [sg.Button('Main Menu'), sg.Button('Output Data'), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, size=(600, 200), finalize=True)


    def make_win6():
        layout = [
            [sg.Text("Graph output, use the navigation controls below to save, zoom in or move the figure left or right.")],
            [sg.T('Controls:')],
            [sg.Canvas(key='controls_cv')],
            [sg.T('Figure:')],
            [sg.Column(
                layout=[
                   [sg.Canvas(key='fig_cv', size=(3500 * 2, 2000))]
                ],
                background_color='#DAE0E6',
                pad=(0, 0)
            )],
            [sg.Button("Export to CSV"), sg.Button("Display Count Graph"), sg.Button("Main Menu"),
                   sg.Button("New Chronology"), sg.Button("Exit")]]
        return sg.Window('Crossdating App', layout, resizable=True, finalize=True, size =(9000,1000))


    sg.set_options(font=('Arial Bold', 16))
    sg.theme('DarkGreen')
    window1, window2, window3, window4, window5, window6 = make_win1(), None, None, None, None, None

    while True:  # Event Loop
        window, event, values = sg.read_all_windows()

        if event == sg.WIN_CLOSED or event == 'Exit':

            window.close()
            break
        elif event == 'Upload CSV file' and not window2:
            window.close()
            window1 = None
            window2 = make_win2()
        elif event == "Import":
            df = helper.read_csv_to_dataframe(values["-IN-"])
            window.close()
            window2 = None
            window3 = make_win3()
        elif event == 'Statistical Method':
            window.close()
            window3 = None
            window4 = make_win4()
        elif event == "Submit":
            window.close()
            window4 = None
            window5 = make_win5()
            df = helper.rename_dataframe(df)
            samples = helper.convert_dataframe_to_list(df)
            window5['-PBAR-'].update(current_count=0)
            window5['-OUT-'].update("Padding stripped from sample...")
            segment_size = int(values['-SEGMENT-'])
            sd = int(values['-STANDDEV-'])
            step = int(values['-OUTLIERS-'])
            nostart = int(values['-STARTDATE-'])
            window5['-PBAR-'].update(current_count=10)
            segments = helper.divide_list_to_segments(samples, segment_size)
            window5['-PBAR-'].update(current_count=20)
            window5['-OUT-'].update(f"Divided sample and master chronology in to {segment_size} sized segments...")
            pairs, stride = statistical_method.matching_pairs(segments)
            window5['-PBAR-'].update(current_count=30)
            window5['-OUT-'].update("Master chronology and Sample pairs have been created...")
            t_vals = statistical_method.t_values(pairs, segment_size)
            window5['-PBAR-'].update(current_count=50)
            window5['-OUT-'].update("Statistical T-values have been calculated...")
            top_contenders = statistical_method.sig_t_val(t_vals, sd, stride, step)
            window5['-PBAR-'].update(current_count=70)
            window5['-OUT-'].update("Statistically significant T-values have been found...")
            start_years = statistical_method.top_pairs(top_contenders, t_vals, samples, nostart)
            window5['-PBAR-'].update(current_count=80)
            window5['-OUT-'].update("Start years for crossdate have been found...")
            for index in range(len(start_years)):
                statistical_method.adding_padding(df, samples, start_years, index)
            window5['-PBAR-'].update(current_count=100)
            window5['-OUT-'].update("Crossdate Complete!")
        if event == 'Output Data':
            window.close()
            window5 = None
            window6 = make_win6()
            plt.plot(df["master_chronology"], color="#40B0A6")
            plt.plot(df['Aligned_0'], color="#E1BE6A")
            plt.legend(['master chronology', 'sample'], fontsize=10)
            plt.ylabel('detrended sample value', fontsize=16)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.xlabel('year', fontsize=16)
            plt.grid(True)
            plt.title("Detrended values of a master chronology and a sample.", fontsize=16)
            plt.figure(1)
            fig = plt.gcf()
            DPI = fig.get_dpi()
            fig.set_size_inches(1000 * 2 / float(DPI), 600 / float(DPI))
            draw_figure_w_toolbar(window6['fig_cv'].TKCanvas, fig, window6['controls_cv'].TKCanvas)
        elif event == "Export to CSV":
            window.close()
            window6 = None
            df.to_csv()
        elif event == 'Main Menu':
            window.close()
            window5 = None
            window3 = make_win3()
        elif event == 'New Chronology':
            window.close()
            window5 = None
            window2 = make_win2()
        elif event == 'Display Count Graph':
            window.close()
            window5 = None
            #Add count graph

        elif event == 'Machine Learing Method':


    window.close()
