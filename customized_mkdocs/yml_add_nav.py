import os
import os.path
import os,sys,shutil
import argparse

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
                
def dfs_writedir(path, depth, file, ignore_set):
    for item in os.listdir(path):
        if item in ignore_set:
            continue
        elif '.git' not in item:
            to_write = " " * 2 * (depth + 1) + "- " + item.split(".")[0] + ': '
            if path == 'docs' and item == "index.md":
                to_write = " " * 2 * (depth + 1) + "- index.md"
            elif path == f'docs/{args.blog_dir}' and item == args.post_dir: 
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
                    
            file.write(to_write + '\n')

            newitem = path +'/'+ item
            if os.path.isdir(newitem):
                dfs_writedir(newitem, depth + 1, file, ignore_set)
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description=(
                "Rewrite an input mkdocs YAML to output, appending a subfolder to "
                "plugins -> blog -> post_dir if present."
            )
        )

    parser.add_argument("--blog_dir", type=str, default="Blog")
    parser.add_argument("--post_dir", type=str, default="Posts")
    parser.add_argument("--doc_dir", type=str, default="docs")
    parser.add_argument("--ignore_set", type=list, default=['.DS_Store', 'assets', 'stylesheets', 'javascript'])
    parser.add_argument("-i", "--input_yml", type=str, default="mkdocs_config/mkdocs_config.yml")
    parser.add_argument("-o", "--output_yml", type=str, default="mkdocs.yml")
    args = parser.parse_args()

    shutil.copyfile(args.input_yml, args.output_yml)

    with open(args.output_yml,"a") as currentfile:  
        currentfile.write("\nnav: " + '\n')       
        dfs_writedir(args.doc_dir, 0, file = currentfile, ignore_set = args.ignore_set)