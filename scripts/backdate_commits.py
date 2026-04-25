import os
import subprocess
from datetime import datetime

def main():
    subprocess.run(["git", "init"])
    
    # Get all non-ignored files
    result = subprocess.run(["git", "ls-files", "-o", "--exclude-standard"], capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    
    if not files or files == ['']:
        print("No files to commit.")
        return

    # Store file along with its exact modification time
    files_with_time = []
    for f in files:
        if not os.path.exists(f):
            continue
        mtime = os.path.getmtime(f)
        files_with_time.append((f, mtime))
        
    # Sort files globally by exact chronological order
    files_with_time.sort(key=lambda x: x[1])
    
    print(f"Total files to commit: {len(files_with_time)}")

    for f, mtime in files_with_time:
        commit_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Add the specific file
        subprocess.run(["git", "add", f])
        
        # We can extract the file name for the commit message
        file_basename = os.path.basename(f)
        commit_msg = f"Add or update {file_basename}"
        
        print(f"Committing {f} at {commit_date}")
        
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = commit_date
        env['GIT_COMMITTER_DATE'] = commit_date
        env['GIT_AUTHOR_NAME'] = "Rahul"
        env['GIT_AUTHOR_EMAIL'] = "rahulbiradar7134@gmail.com"
        env['GIT_COMMITTER_NAME'] = "Rahul"
        env['GIT_COMMITTER_EMAIL'] = "rahulbiradar7134@gmail.com"
        
        subprocess.run(["git", "commit", "-m", commit_msg], env=env)
        
    print("All individual file commits created successfully.")

if __name__ == "__main__":
    main()
