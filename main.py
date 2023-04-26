import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from collections import Counter
from os import listdir

import helper
import machine_learning_method
import statistical_method
import cProfile


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


if __name__ == "__main__":

    def add_labels(x, y):
        for label in range(len(x)):
            plt.text(label, y[label] // 2, y[label], ha='center')


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
            [sg.Text("Welcome to the crossdating application, you will be required to input\nseveral decision "
                     "about your "
                     "data in order this program to complete."), sg.Text('      ', k='-OUTPUT-')],
            [sg.Button('Upload CSV file'), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(605, 100), finalize=True)


    def make_win2():
        layout = [
            [sg.T(
                "Please upload a CSV file containing a 3 colums:\n1) The index of years for a master chronology\n2) "
                "The master chronology correctly indexed by the years column\n3) The sample chronology, this can be "
                "added at any point in the \ncolumn and is required to have padding so all the columns are the same "
                "length ")],
            [sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],
            [sg.Button("Import"), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(700, 230), finalize=True)


    def make_win3():
        layout = [[sg.Text("Below are two methods for crossdating your data, a machine learning \nmethod and a"
                           "statistical method. Both will give further options to save the \nresults or graph them.")],
                  [sg.Button("Statistical Method"),
                   sg.Button("Machine Learning Method")]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(605, 120), finalize=True)


    def make_win4():
        labels = ('Segment size the sample is split into (default: 10 years):',
                  'The number of standard deviations from the mean that is classified as an outlier (default: 3 '
                  'standard diviations):',
                  'Number of consecutive outliers considered significant (default: 8 consecutive outliers):',
                  'Number of start years the program outputs (default: 1, the highest likely year):')
        keys = ('-SEGMENT-', '-STANDDEV-', '-OUTLIERS-', '-STARTDATE-')
        default = ['10', '3', '8', '1']
        size = max(map(len, labels))
        layout = [[sg.Text("Statistical Method based on pairwise lead lag analysis.\n You will be asked to to make "
                           "decisions that will effect the ability of the program to crossdate the sample.\n At each "
                           "step a default will be suggested.\n")],
                  [
                      [sg.Text(labels[i], size=size), sg.Input(default[i], key=keys[i].split()[0])]
                      for i in range(len(labels))] + [[sg.Push(), sg.Button("Submit")]],
                  [sg.Text(
                      "WARNING: Please make sure the segment size x  consecutive outliers is less than the length of the sample chronology.")]]

        return sg.Window('Crossdating App', layout, resizable=True, size=(1200, 250), finalize=True)


    def make_win5():
        layout = [[sg.Text("Computation is occuring please wait.")],
                  [sg.ProgressBar(100, orientation='h', expand_x=True, key='-PBAR-', size=(20, 20))],
                  [sg.Text('', key='-OUT-', enable_events=True, justification='center', expand_x=True)],
                  [sg.Button('Main Menu'), sg.Button('Output Data'), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(600, 160), finalize=True)


    def make_win6():
        layout = [
            [sg.Text(
                "Graph output, use the navigation controls below to save, zoom in or move the figure left or right.")],
            [sg.T('Controls:')],
            [sg.Canvas(key='controls_cv')],
            [sg.T('Figure:')],
            [sg.Column(
                layout=[
                    [sg.Canvas(key='fig_cv', size=[3500 * 2, 3800])]
                ],
                pad=(0, 0),
            )],
            [sg.Button("Export to CSV"), sg.Button("Display Bar Chart of Start years"), sg.Button("Main Menu"),
             sg.Button("New Chronology"), sg.Button("Exit")]]
        return sg.Window('Crossdating App', layout, resizable=True, finalize=True, size=(9000, 1200))


    def make_win7():
        layout = [
            [sg.Text("Thank you for using this crossdating application, your output has been saved.\n"
                     "If you would like to continue to please use the buttons below.")],
            [sg.Button("New Chronology"),
             sg.Button("Statistical Method"),
             sg.Button("Machine Learning Method")],
            [sg.Button("Exit")]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(610, 128), finalize=True)


    def make_win8():
        layout = [
            [sg.Text(
                "Graph output, use the navigation controls below to save, zoom in or move the figure left or right.")],
            [sg.T('Controls:')],
            [sg.Canvas(key='controls_cv2')],
            [sg.T('Figure:')],
            [sg.Column(
                layout=[
                    [sg.Canvas(key='fig_cv2', size=(3700 * 2, 3500))]
                ],
                background_color='#DAE0E6',
                pad=(0, 0)
            )],
            [sg.Button("Export to CSV"), sg.Button("Main Menu"),
             sg.Button("New Chronology"), sg.Button("Exit")]]
        return sg.Window('Crossdating App', layout, resizable=True, finalize=True, size=(9000, 1000))


    def make_win9():
        layout = [
            [sg.T(
                "Please upload a folder of CSV files to train the method,\nplease do not use the "
                "testing data.\n The CSV files should contain 3 columns given below:\n1) The index of years for a "
                "master chronology\n2) The master chronology correctly "
                "indexed by the years column\n3) The sample chronology, this can be added at any point in the "
                "\ncolumn and is required to have padding so all the columns are the same length")],
            [sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FolderBrowse(key='-FOLDER-')],
            [sg.Button("Import Training data"), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(800, 260), finalize=True)


    def make_win10():
        layout = [[sg.Text("Computation is occuring please wait. This may take up to 15 minutes.")],
                  [sg.ProgressBar(100, orientation='h', expand_x=True, key='-PBAR2-', size=(20, 20))],
                  [sg.Text('', key='-OUT2-', enable_events=True, justification='center', expand_x=True)],
                  [sg.Button('Main Menu'), sg.Button('Output Data'), sg.Button('Exit')]]
        return sg.Window('Crossdating App', layout, resizable=True, size=(600, 160), finalize=True)


    sg.set_options(font=('Arial Bold', 16))
    sg.theme('DarkGreen')
    window1, window2, window3, window4, window5 = make_win1(), None, None, None, None
    window6, window7, window8, window9, window10 = None, None, None, None, None
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
            top_start_years, start_years = statistical_method.top_pairs(df, top_contenders, t_vals, samples, nostart)
            window5['-PBAR-'].update(current_count=80)
            window5['-OUT-'].update("Start years for crossdate have been found...")
            output = df.copy()
            for index in range(len(top_start_years)):
                output = statistical_method.adding_padding(df, samples, top_start_years, index, output)
            window5['-PBAR-'].update(current_count=100)
            window5['-OUT-'].update("Crossdate Complete!")
        if event == 'Output Data':
            window.close()
            window5 = None
            window6 = make_win6()
            plt.figure()
            plt.plot(output["master_chronology"], color="#40B0A6")
            plt.plot(output['Aligned_0'], color="#E1BE6A")
            plt.legend(['master chronology', 'sample'], fontsize=8)
            plt.ylabel('detrended sample value', fontsize=8)
            plt.yticks(fontsize=8)
            plt.xticks(fontsize=8)
            plt.xlabel('year', fontsize=8)
            plt.grid(True)
            plt.title("Detrended values of a master chronology and a sample.", fontsize=10)
            fig = plt.gcf()
            DPI = fig.get_dpi()
            fig.set_size_inches(1150 * 2 / float(DPI), 830 / float(DPI))
            draw_figure_w_toolbar(window6['fig_cv'].TKCanvas, fig, window6['controls_cv'].TKCanvas)
        elif event == "Export to CSV":
            output.to_csv('crossdating_output.csv')
            window.close()
            window6 = None
            window7 = make_win7()
        elif event == 'Main Menu':
            window.close()
            window5 = None
            window3 = make_win3()
        elif event == 'New Chronology':
            window.close()
            window5 = None
            window2 = make_win2()
        elif event == 'Display Bar Chart of Start years':
            plt.clf()
            window.close()
            window5 = None
            window8 = make_win8()
            # Get the Keys and store them in a list
            labels = list([start_years.most_common()[item][0] for item in
                           range(len(start_years.most_common(15)))])
            labels = [str(label) for label in labels]
            # Get the Values and store them in a list
            values = list([start_years.most_common()[item][1] for item in
                           range(len(start_years.most_common(15)))])
            plt.figure()
            plt.title("Bar chart of all possible start years from crossdating program.", fontsize=10)
            plt.xlabel("Start year", fontsize=6)
            plt.ylabel("Counts", fontsize=8)
            plt.bar(labels, values, color="#40B0A6")
            add_labels(labels, values)
            fig2 = plt.gcf()
            DPI2 = fig2.get_dpi()
            fig2.set_size_inches(1150 * 2 / float(DPI2), 790 / float(DPI2))
            draw_figure_w_toolbar(window8['fig_cv2'].TKCanvas, fig2, window8['controls_cv2'].TKCanvas)
        elif event == 'Machine Learning Method':
            window.close()
            window6 = None
            window9 = make_win9()
        elif event == 'Import Training data':
            window.close()
            window9 = None
            window10 = make_win10()
            df = helper.rename_dataframe(df)
            i = 1
            training_data = []
            filenames = listdir(values["-FOLDER-"])
            folder = [filename for filename in filenames if filename.endswith('.csv')]
            folder = [str(values["-FOLDER-"]) + '/' + str(filename) for filename in filenames]
            for data in folder:
                name = str('df') + str(i)
                try:
                    name = helper.read_csv_to_dataframe(data)
                    training_data.append(name)
                except:
                    continue
                i += 1
            window10['-PBAR2-'].update(current_count=0)
            window10['-OUT2-'].update("Configuring the training dataset...")
            x, y = machine_learning_method.setting_up_training_data(training_data)
            strat = pd.Series(y)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=strat)
            mlp = MLPClassifier(hidden_layer_sizes=(300, 100,), alpha=0.001, max_iter=7000)
            window10['-PBAR2-'].update(current_count=25)
            window10['-OUT2-'].update("Training the first model...")
            mlp.fit(x_train, y_train)
            window10['-PBAR2-'].update(current_count=30)
            window10['-OUT2-'].update("Running first model...")
            mlp.fit(x_train, y_train)
            test_x, test_y = machine_learning_method.training_data_for_first_mlp(df)
            y_pred = mlp.predict(x_test)
            y_test_pred = mlp.predict(test_x)
            model_1_output = pd.DataFrame()
            model_1_output['Input'] = test_x
            model_1_output['Output'] = y_test_pred
            model_1_training_output = pd.DataFrame()
            model_1_training_output['Input'] = x_test
            model_1_training_output['Output'] = y_pred
            true_model_output = model_1_output.loc[model_1_output['Output'] == 1]
            true_training_model_output = model_1_training_output.loc[model_1_training_output['Output'] == 1]
            inputs = true_model_output['Input'].to_numpy()
            inputs = inputs.tolist()
            input2 = []
            for i in range(len(inputs)):
                input2.append(inputs[i][0:100])
            input_no_dup = [i for n, i in enumerate(input2) if i not in input2[:n]]
            input_pairs = machine_learning_method.create_pairs([df], input_no_dup)
            training_inputs = true_training_model_output['Input'].to_numpy()
            training_inputs = training_inputs.tolist()
            train_inputs2 = []
            for i in range(len(training_inputs)):
                train_inputs2.append(training_inputs[i][0:100])
            train_input_no_dup = [i for n, i in enumerate(train_inputs2) if i not in train_inputs2[:n]]
            train_input_pairs = machine_learning_method.create_pairs(training_data, train_input_no_dup)
            correct_pair_training = machine_learning_method.correct_training_pairs(training_data)
            correct_pair = machine_learning_method.correct_pairs(df)
            binary_training = machine_learning_method.training_data_for_second_mlp(train_input_pairs,
                                                                                   correct_pair_training)
            x2 = train_input_pairs
            y2 = binary_training
            x2 = pd.Series(x2)
            y2 = pd.Series(y2)
            window10['-PBAR2-'].update(current_count=40)
            window10['-OUT2-'].update("Extracting data for the second model...")
            x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.5)
            training_set = pd.concat([x2_train, y2_train], axis=1)
            match = training_set[training_set.iloc[:, 1] == 1]
            no_match = training_set[training_set.iloc[:, 1] == 0]
            undersample = resample(no_match, replace=True, n_samples=len(match))
            undersample_train = pd.concat([match, undersample])
            undersample_train.iloc[:, 1].value_counts(normalize=True)
            undersample_x_train = np.array(undersample_train.drop(1, axis=1))
            undersample_y_train = undersample_train[1]
            undersample_x_train_list = []
            for i in range(len(undersample_x_train)):
                if len(undersample_x_train[i][0]) == 20:
                    undersample_x_train_list.append(undersample_x_train[i][0])
                else:
                    continue
            x_test_list = np.array(x2_test, dtype=list)
            x2_test_final = []
            for i in range(len(x_test_list)):
                if len(x_test_list[i]) == 20:
                    x2_test_final.append(x_test_list[i])
                else:
                    continue
            mlp2 = MLPClassifier(hidden_layer_sizes=(164, 160, 120,), alpha=0.0001, max_iter=5000)
            window10['-PBAR2-'].update(current_count=50)
            window10['-OUT2-'].update("Training the second model...")
            mlp2.fit(undersample_x_train_list, undersample_y_train)
            window10['-PBAR2-'].update(current_count=60)
            window10['-OUT2-'].update("Running second model...")
            y2_pred = mlp2.predict(x2_test_final)
            y2_pred_test = mlp2.predict(input_pairs)
            model_2_training_output = pd.DataFrame()
            model_2_training_output['Input'] = x2_test_final
            model_2_training_output['Output'] = y2_pred
            model_2_output = pd.DataFrame()
            model_2_output['Input'] = input_pairs
            model_2_output['Output'] = y2_pred_test
            true_model_training_output2 = model_2_training_output.loc[model_2_training_output['Output'] == 1]
            true_model_output2 = model_2_output.loc[model_2_output['Output'] == 1]
            training_inputs2 = true_model_training_output2['Input'].to_numpy()
            inputs2 = true_model_output2['Input'].to_numpy()
            training_inputs3 = training_inputs2.tolist()
            inputs3 = inputs2.tolist()
            training_correct = machine_learning_method.training_data_for_third_mlp(training_inputs2,
                                                                                   correct_pair_training)
            window10['-PBAR2-'].update(current_count=65)
            window10['-OUT2-'].update("Extracting data for the third model...")
            x3 = pd.Series(training_inputs3)
            y3 = pd.Series(training_correct)
            x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.5)
            training_set3 = pd.concat([x3_train, y3_train], axis=1)
            match3 = training_set3[training_set3.iloc[:, 1] == 1]
            no_match3 = training_set3[training_set3.iloc[:, 1] == 0]
            undersample3 = resample(no_match3, replace=True, n_samples=len(match3))
            undersample_train3 = pd.concat([match3, undersample3])
            undersample_train3.iloc[:, 1].value_counts(normalize=True)
            undersample_x_train3 = np.array(undersample_train3.drop(1, axis=1))
            undersample_y_train3 = undersample_train3[1]
            undersample_x_train_list3 = []
            for i in range(len(undersample_x_train3)):
                if len(undersample_x_train3[i][0]) == 20:
                    undersample_x_train_list3.append(undersample_x_train3[i][0])
                else:
                    continue
            x_test_list3 = np.array(x3_test, dtype=list)
            x_test_final3 = []
            for i in range(len(x_test_list3)):
                if len(x_test_list3[i]) == 20:
                    x_test_final3.append(x_test_list3[i])
                else:
                    continue
            mlp3 = MLPClassifier(hidden_layer_sizes=(525,), alpha=0.0001, learning_rate='invscaling', max_iter=600)
            window10['-PBAR2-'].update(current_count=70)
            window10['-OUT2-'].update("Training the third model...")
            mlp3.fit(undersample_x_train_list3, undersample_y_train3)
            window10['-PBAR2-'].update(current_count=75)
            window10['-OUT2-'].update("Running third model...")
            y3_pred = mlp3.predict(x_test_final3)
            output3 = mlp3.predict(inputs3)
            model_3_output = pd.DataFrame()
            model_3_output['Input'] = inputs3
            model_3_output['Output'] = output3
            true_model3_output = model_3_output.loc[model_3_output['Output'] == 1]
            all_contender = true_model3_output['Input'].to_numpy()
            start_year = []
            samples = helper.convert_dataframe_to_list(df)
            window10['-PBAR2-'].update(current_count=85)
            window10['-OUT2-'].update("Output found, adding to dataset...")
            for i in range(len(all_contender.tolist())):
                (master_seg, sample_seg) = all_contender[i][0:9], all_contender[i][10:20]
                start_year.append(
                    df.first_valid_index() + samples[0].index(master_seg[0]) - samples[1].index(sample_seg[0]))
            start_years = Counter(start_year)
            output = df.copy()
            output = machine_learning_method.adding_padding(df, samples, start_years, 1, output)
            window10['-PBAR2-'].update(current_count=100)
            window10['-OUT2-'].update("Crossdate Complete!")

    window.close()
