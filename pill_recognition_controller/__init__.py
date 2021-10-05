import logging
import os
import sys
import pandas as pd
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from cv2 import imread

UPLOAD_FOLDER = '/usr/src/app/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
IMAGE_SIZE = 224
IMAGE_SIZE = 224
CHANNELS = 3

api = Flask(__name__)
api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if __name__ == '__main__':
    api.run(debug=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_table_style():
    style = '''
        <style>
        .styled-table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: Arial;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
        .styled-table tbody tr.active-row {
            font-weight: bold;
            color: #009879;
        }
        </style>    
    '''
    return style


def prepare_output(prediction):
    file_path = '/usr/src/app/models/mappings.xlsx'
    mappings_df = pd.read_excel(file_path)
    for index, row in mappings_df.iterrows():
        mappings_df.iloc[[index], [0]] = int(row['Classification'])

    predictions = [0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00]
    mappings_df["prediction"] = predictions
    print(mappings_df, file=sys.stderr)

    tbl_body = ''
    tbl_begin = '<table class="styled-table">'
    tr_header = '''
    <thead>
        <tr>
        <th>Classification</th>
        <th>NDC</th>
        <th>Label</th>
        <th>ProprietaryName</th>
        <th>DosageForm</th>
        <th>RouteName</th>
        <th>CompanyName</th>
        <th>DrugStatus</th>
        <th>Probability (%)</th>
        </tr>
    </thead>
    <tbody>       
    '''
    tbl_end = '</tbody> </table>'
    tbl_body = tbl_begin + tr_header

    for r in prediction:
        ctr = 0
        for i in r:
            i = float(i) * 100
            mappings_df.iloc[[ctr], [8]] = i
            ctr += 1

    sorted_df = mappings_df.sort_values(by=['prediction', 'Classification'], ascending=(False, True))
    print(sorted_df, file=sys.stderr)

    for index, row in sorted_df.iterrows():
        clazz_value = row['prediction']
        if clazz_value >= 0.5:
            row_str = '''
                <tr class="active-row">
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
            '''.format(row['Classification'],
                       row['NDC'],
                       row['Label'],
                       row['ProprietaryName'],
                       row['DosageForm'],
                       row['RouteName'],
                       row['CompanyName'],
                       row['Status'],
                       "{:.7f}".format(clazz_value))
        else:
            row_str = '''
            <tr>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{}</td>
            </tr>
        '''.format(row['Classification'],
                   row['NDC'],
                   row['Label'],
                   row['ProprietaryName'],
                   row['DosageForm'],
                   row['RouteName'],
                   row['CompanyName'],
                   row['Status'],
                   "{:.7f}".format(clazz_value))

        tbl_body += row_str

    tbl_body += tbl_end
    return tbl_body


def process_model(model, filename):
    logging.debug('process_mode method called with: ', model)
    if 'resnet50' == model:
        model_filepath = '/usr/src/app/models/resNet50_custom_fin_model_v2.h5'
        logging.debug('loading trained model from ', model_filepath)
        model = load_model(model_filepath)
        img_array = imread(os.path.join(api.config['UPLOAD_FOLDER'], filename))
        img_array = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        prediction = model.predict([img_array])

        return '''
            <html>
                <head>
                    <title>
                        upload pill image
                    </title>
                    <style>{}</style>
                </head>
                <body>
                    <h1>Received your pill image file</h1>
                    <h1>PREDICTION:</h1>
                    <p>{}</p>
                    <a href="javascript:history.back()">Try another image</a>
                </body>
            </html>            
            '''.format(get_table_style(), prepare_output(prediction))
    elif 'custom' == model:
        model_filepath = '/usr/src/app/models/25_custom_model_final.h5'
        logging.debug('loading trained model from ', model_filepath)
        model = load_model(model_filepath)
        img_array = imread(os.path.join(api.config['UPLOAD_FOLDER'], filename))
        img_array = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        prediction = model.predict([img_array])

        return '''
            <html>
                <head>
                    <title>
                        upload pill image
                    </title>
                    <style>{}</style>
                </head>
                <body>
                    <h1>Received your pill image file</h1>
                    <h1>PREDICTION:</h1>
                    <p>{}</p>
                    <a href="javascript:history.back()">Try another image</a>
                </body>
            </html>
            '''.format(get_table_style(), prepare_output(prediction))


@api.route('/<model>', methods=['GET', 'POST'])
def get_classification(model):
    logging.debug('get_classification method called with: ', request.method)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            logging.debug('saving file to: ', os.path.join(api.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(api.config['UPLOAD_FOLDER'], filename))
            return process_model(model, filename)

    return '''
    <!doctype html>
       <title>Run CNN Model</title>
       <body>
          <form action = "http://localhost:9595/{}" method = "POST" 
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>
       </body>
    </html>
    '''.format(model)
