import re
import os
import requests
from multiprocessing import Pool, cpu_count
from datetime import datetime

def request_link(args):
    name, url = args
    session = requests.Session()
    # fake session
    session.headers.update(
        {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
    )
    
    response = session.head(url, allow_redirects=True)
    if response.status_code != 200:
        response = session.get(url, allow_redirects=True)

    return (name, url, response.status_code)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    readme_file = os.path.join(current_dir, "../readme.md")
    pool = Pool(cpu_count() - 1)
    
    with open(os.path.join(current_dir, "../reports.md"), 'w') as outfile:
        outfile.write("## Report\n\n")
        outfile.write(f"Date: {datetime.utcnow()}\n\n")
        outfile.write("| URL         | Status |\n")
        outfile.write("| ----------- | ----------- |\n")
        
        with open(readme_file) as f:
            readme_content = f.read()
            links = [
                (m.group(1), m.group(2))
                for m in re.finditer(
                    r"\[(.+)\]\((https?[^\)]+)\)", readme_content, flags=re.MULTILINE
                )
            ]
            
            for name, link, status_code in pool.imap_unordered(request_link, links):
                status = "OK" if status_code == 200 else f"Failed({status_code})"
                outfile.write(f"|[{name}]({link})|`{status}`|\n")
                print(status_code, name, link)
            
    pool.close()
    pool.join()
    