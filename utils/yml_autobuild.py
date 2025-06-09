import os
import os.path
import os,sys,shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="published", choices=["published", "preview"])
args = parser.parse_args()

def is_dir_empty(path, depth):
    for item in os.listdir(path):
        if os.path.isdir(path + '/' + item):
            pass
        else:
            return False
        newitem = path + '/' + item
        if os.path.isdir(newitem):
            is_dir_empty(newitem, depth + 1)
    return True

def get_first_line(path):
    with open(path, 'r') as f:  
        lines = f.readlines()  
        first_line = lines[0]  
    
    return first_line[2:-1]
                
def dfs_writedir(path, depth, file):
    for item in os.listdir(path):
        if item in ignore_set:
            continue
        elif '.git' not in item:
            to_write = " " * 2 * (depth + 1) + "- " + item.split(".")[0] + ': '
            if path == 'docs' and item == "index.md":
                to_write = " " * 2 * (depth + 1) + "- index.md"
            elif path == 'docs/Blog' and item == 'Posts': 
                continue  
            elif path == 'docs' and item == 'index.md':
                continue
            elif len(item.split(".")) > 1:
                if item.split(".")[1] == 'md':
                    newpath = ''
                    if path[5:] != '':
                        newpath = path[5:] + '/'
                    to_write += newpath + item
                else:
                    continue
            elif len(item.split(".")) == 1:
                    pass
                    
            currentfile.write(to_write + '\n')

            newitem = path +'/'+ item
            if os.path.isdir(newitem):
                dfs_writedir(newitem, depth +1, file)
                
global_path = "docs"
ignore_set = ['.DS_Store', 'assets', 'stylesheets', 'javascript']
dirc_set = os.listdir(global_path)
if args.mode == "published":
    shutil.copyfile('./utils/base_mkdocs_published.yml', './mkdocs.yml')
else:
    shutil.copyfile('./utils/base_mkdocs_preview.yml', './mkdocs.yml')

with open("mkdocs.yml","a") as currentfile:  
    currentfile.write("nav: " + '\n')       
    dfs_writedir('docs', 0, file = currentfile)