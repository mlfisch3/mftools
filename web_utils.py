import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.loader import ItemLoader
import re
import lxml.html
import requests
from bs4 import BeautifulSoup

#  download_from_url(url, dPath=None, file_extension='.pdf')
#  download_links(links, dPath=None):
#  find_links(url, file_extension='.pdf')
#  strip_html(data):
#  strip_html(data):
#  strip_html_doc(data):
#  table_from_url(url, n_columns=None):
#  parse_tr(trow):
#  get_index_components(url_base=URL_BASE, index_name=INDEX_NAME):

# pdfs = [link.get('href') for link in a_list if link.has_attr('href')]
# pdfs = [link for link in pdfs if link.split('.')[-1]=='pdf']
# links = [f'{prefix}{link}' if 'https' not in link else f'{link}' for link in pdfs]
# with open('C:/CODE/TMP/links.txt', 'w') as fout:
    # for link in links:
    #     fout.write(link.strip()+'\n')
    
# use subprocess function to run wget from system prompt


class JokeItem(scrapy.Item):
	joke_text = scrapy.Field()

class JokesSpider(scrapy.Spider):
    name='jokes'
    
    start_urls= [ 'http://www.laughfactory.com/jokes/family-jokes' ]
    
    def parse(self, response):
        for joke in response.xpath("//div[@class='jokes']"):
            yield { 'joke_text': joke.xpath(".//div[@class='joke-text']/p").extract_first() }
            
        next_page = response.xpath("//li[@class='next']/a/@href").extract_first()
        if next_page is not None:
            next_page_link = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link, callback=self.parse)


### remove html tags from string ###

#using regex (slow):


def strip_html(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

# using lxml.html (faster, preferred):


def strip_html(data):
    text = lxml.html.fromstring(str(data)).text_content()
    return text.encode('ascii', 'ignore').decode('utf-8')

# def strip_html(data):
# 	return lxml.html.fromstring(data).text_content()


def strip_html_doc(data):
	page = lxml.html.document_fromstring(data)
	return page.cssselect('body')[0].text_content()



def table_from_url(url, n_columns=None):
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    table_headers = soup.find_all('th')
    table_data = soup.find_all('td')

    table_headers = list(map(strip_html, table_headers))
    table_data = list(map(strip_html, table_data))
    col_names = list(map(remove_newline, table_headers))

    if n_columns is None:
        n_columns = len(table_headers)

    col_names = col_names[:n_columns]  # remove extra 'Cobalt (Co)'

    data = wrap_list(table_data, n_columns)
    steels_ns_df = pd.DataFrame(data, columns=col_names)


###################### simple file downloader  #############################

def find_links(url, file_extension='.pdf')
    #url ='https://www.math.nyu.edu/~kohn/pde_finance.html'
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    hrefs = soup.find_all('i')
    pattern = r'(a href\=\")(.+)(\")'
    links = []
    for href in hrefs:
        link = re.search(pattern, str(href)).group(2)
        if re.search('.pdf', link):
            links.append(link)

    return links


def download_links(links, dPath=None):

    if not dPath:
        dPath='.'
    
    for link in links:
        fName=link.split('/')[-1]
        fPath=os.path.join(dPath, fName)
        request.urlretrieve(target_file, filename=fPath)        
        print(fPath)


def download_from_url(url, dPath=None, file_extension='.pdf')

    links = find_links(url, file_extension=file_extension)
    download_links(links, dPath=dPath)
    count = len(links)
    print(f'{count} files were saved to local directory: {dPath}')


URL_BASE=r'https://www.slickcharts.com'
INDEX_NAME='dowjones'


def parse_tr(trow):
    line = trow.text.encode('ascii', 'ignore').decode('utf-8')
    pattern = r'([\n]+)'
    return re.sub(pattern, '|', line).split('|')[1:-1]

def get_index_components(url_base=URL_BASE, index_name=INDEX_NAME):
    
    """
    url_base:         'https://www.slickcharts.com'
    index_name:  'dowjones' or 'sp500' or 'nasdaq100'
    """
    
    user_agent_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    proxy = None

    url = parse.urljoin(url_base, index_name)
    r = requests.get(url=url, proxies=proxy, headers=user_agent_headers)
    s = BeautifulSoup(r.text, 'lxml')

    trows = s.tbody.find_all('tr')

    components = [parse_tr(trow) for trow in trows]
    
    return pd.DataFrame(components, columns=['id','name','ticker','weight','price','return_abs','return_pct'])


import pandas as pd
import re

def extract_url_re(href):
    rgx = r'(url\?q\=)(\S+)(&sa\=)'
    return re.search(rgx, href)

def get_target_info(target_url):

    r=requests.get(target_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(r.text, 'html.parser')
    hrefs = soup.find_all("a", href=lambda x: x and "." in os.path.basename(x))
    files = list(map(lambda x: x["href"], hrefs))
    
    if target_url[:30]=='https://www.google.com/search?':
        urls = [y.group(2) for y in (extract_url_re(x) for x in files) if y is not None]
        urls = [x for x in urls if 0 < len(x.split('.')[-1]) < 5]
        filenames = list(map(lambda x: x.split('/')[-1], urls))
        filenames = list(map(lambda x: x.replace(r'%20','_').replace('__','_-_'), filenames))
    else:   
        filenames = list(map(lambda x: x.replace(r'%20','_').replace('__','_-_'), files))
        urls = list(map(lambda x: request.urljoin(target_url, x) , files))
        
    filetypes = list(map(lambda x: x.split('.')[-1] , filenames))

    return pd.DataFrame({'File':filenames, 'URL': urls, 'Type':filetypes})


def download_by_urlretrieve(filenames, urls, dName, delay_lo, delay_hi, delay=False):
        
    target_count = len(urls)
    not_downloaded = []
    successful_files = []
    successful_urls = []

    for i, (filename, target_file) in stqdm(enumerate(zip(filenames, urls))):
        print('Downloading (file {} of {} ): {}  ...'.format(i+1, target_count, target_file))
        'Downloading (file {} of {} ): {}  ...'.format(i+1, target_count, target_file)
        fPath = os.path.join(dName, filename)
        fPath = fPath.replace(' ', '_')
        try:
            request.urlretrieve(target_file, filename=fPath)
            successful_files.append(filename)
            successful_urls.append(target_file)
        except:
            print("  ╚═► Download failed: {}".format(target_file))
            target_file_ = target_file.replace(' ', '%20')
            if target_file_ == target_file:
                not_downloaded.append(target_file)
                continue
            else:
                print('  Downloading (file {} of {} ): {}  ...'.format(i+1, target_count, target_file_))
                try:
                    request.urlretrieve(target_file_, filename=fPath)
                    successful_files.append(filename)
                    successful_urls.append(target_file_)
                except:
                    print("  ╚═► Download failed: {}".format(target_file_))
                    not_downloaded.append(target_file_)
                    continue
        
        if delay:    
            delta = random.randint(delay_lo,delay_hi)
            time = timeit.timeit('time.sleep(0.01)', number=delta)
            print('[',delta, '] sleeping for ', time,' seconds...')
        
    source_info_file_path = os.path.join(dName, "source_info.psv")
    with open(source_info_file_path, 'w') as source_info_file:
        for j, (fname, furl) in enumerate(zip(successful_files, successful_urls)):
            source_info_file.write('|'.join([str(j), fname, furl]) + '\n')

    print(os.lstat(source_info_file_path))
    return not_downloaded


def download_by_wget(urls, dName):

    from shutil import move

    if len(urls) > 0:
        
        # create file containing one target url per line
        with open('urls.txt', 'w') as fout:
            for url in urls:
                fout.write(url + '\n')

        command="wget -i urls.txt --random-wait -P {}".format(dName)

        try:
            # run command in subprocess
            check_output(command)

        except CalledProcessError as e:
            called_process_error = f'exited with error\nreturncode: {e.returncode}\ncmd: {e.cmd}\noutput: {e.output}\nstderr: {e.stderr}'
            wget_failed = True
            message = f'Download was denied'

        move('urls.txt', os.path.join(dName, 'urls.txt'))


def download_to_archive(filenames, urls, delay_lo=30, delay_hi=120, delay=False):
        #wrapper function that 1st applies download_by_urlretrieve(), then applies download_by_wget() on anything not retrieved, saves downloaded files to zip archive, then deletes everything but the zip archive

        from io_utils import create_temporary_directory, zip_dir
        from shutil import rmtree

        temp_dir = create_temporary_directory()

        # Download files to server
        not_downloaded = download_by_urlretrieve(filenames, urls, temp_dir, delay_lo, delay_hi, delay)

        # Try to get any files missed in 1st download attempt
        if len(not_downloaded) > 0:
            print("\n ►►► {} files were not downloaded.  Attempting alternate method ...\n".format(len(not_downloaded)))
            download_by_wget(not_downloaded, temp_dir)

        # Copy downloaded files into compressed archive file
        zip_filename, count_downloaded = zip_dir(temp_dir)
        
        # remove temporary directory 
        rmtree(temp_dir, ignore_errors=True)

        return zip_filename, count_downloaded

