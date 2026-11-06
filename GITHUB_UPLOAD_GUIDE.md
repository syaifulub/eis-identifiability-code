# Publish to GitHub

Suggested repository name: `eis-identifiability-code` (already referenced in `CITATION.cff`).

## Upload using GitHub web interface
1. Create a **new public or private repository** at github.com/new. Do not initialize with README, .gitignore, or license, since these files are included here.
2. Extract `EIS_GitHub_Ready.zip` locally and open its single top-level directory.
3. In the empty repository choose **uploading an existing file**, then upload the *contents* of the extracted directory (including hidden `.github/` and `.gitignore` where supported). For nested folders, the command-line method below is more reliable.

## Recommended: command-line upload
From inside the extracted repository directory:

```bash
git init
git branch -M main
git add .
git commit -m "Initial reproducible research release"
git remote add origin https://github.com/YOUR_USERNAME/eis-identifiability-code.git
git push -u origin main
```

Replace `YOUR_USERNAME` and ensure the remote repository already exists. Authentication is handled by GitHub (browser credential manager or personal access token); do not store passwords/tokens in this repository.

## Before making it public
- Check attribution and license permissions for every supplied dataset and included example.
- Confirm coauthors approve sharing and the journal's data/code policies.
- Recheck `CITATION.cff` repository URL, manuscript status, and authors.
- Run `python -m pytest eisvarpro/tests -q` after installation.
- Inspect all tracked files for confidential data or credentials.
