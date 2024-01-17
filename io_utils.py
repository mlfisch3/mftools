import os

def get_platform_dir_slash(input_file_path):
    OS_NAME = os.name
    if OS_NAME == 'posix':
        input_file_name = input_file_path.split('/')[-1]
    elif OS_NAME == 'nt':
        input_file_name = input_file_path.split('\\')[-1]


def path2tuple(path):
    '''    
    recursively call os.path.split 
    return path components as tuple, preserving hierarchical order

    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> path2tuple(newdir)
    ('C:\\', 'temp', 'subdir0', 'subdir1', 'suubdir2')
          

    '''
    (a,b) = os.path.split(path)
    if b == '':
        return a,
    else:
        return *path2tuple(a), b

def mkpath(path):
    '''
    Similar to os.mkdir except mkpath also creates implied directory structure as needed.

    For example, suppose the directory "C:\\temp" is empty. Build the hierarchy "C:\\temp\\subdir0\\subdir1\\subdir2" with single call:
    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> mkpath(newdir)        

    '''
    u = list(path2tuple(path))    
    pth=u[0]

    for i,j in enumerate(u, 1):
        if i < len(u):
            pth = os.path.join(pth,u[i])
            if not any([os.path.isdir(pth), os.path.isfile(pth)]):
                os.mkdir(pth)


def create_temporary_directory():
    
    temp_dir = os.path.join(os.getcwd(), f'TMP{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    if not os.path.isdir(temp_dir):
        mkpath(temp_dir)

    return temp_dir

import zipfile

def zip_dir(dir_path, zip_filename=None, keep_abs_paths=False):
    '''
    create zip file containing all files in dir_path.  
    directory structure is not preserved unless keep_abs_paths is True

    '''

    if zip_filename is None:
        zip_filename = f'archive_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'

    with zipfile.ZipFile(zip_filename,'w') as zip_file:
        if keep_abs_paths:
            count = len([zip_file.write(os.path.join(dir_path,f)) for f in stqdm(os.listdir(dir_path))])
        else:
            count = len([zip_file.write(os.path.join(dir_path,f), os.path.basename(f)) for f in os.listdir(dir_path)])

    return zip_filename, count    