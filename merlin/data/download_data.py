from merlin.utils import download_file


def download_sample_data(data_dir):
    print("Downloading sample data to {}".format(data_dir))
    file_path = download_file(
        #从哪下载
        repo_id="stanfordmimi/Merlin",    #Hugging Face关于repo_id的命名规则：组织名(或用户名)/仓库名；这里是斯坦福 MIMI 实验室的 Merlin 仓库 
        #下载啥
        filename="image1.nii.gz",         #image1.nii.gz：举例，根据下载文件不同修改
        #下载到哪
        local_dir=data_dir,
    )
    #file_path：load_dir+filename
    return file_path
