# Git new branch and best practices steps

```bash
# Step 1 — Create Your Branch
# Make sure you start from latest main
git checkout main #check if on branch main, if not switch to it
git pull origin main #pull latest main to your local

# Create your new branch and switch to it
git checkout -b feature/branch_name  # create and switch to new branch, name it something descriptive - do only once when creating the branch
git push -u origin feature/branch_name #push your new branch to GitHub and set upstream only do once when creating the branch
```

# Step 2 — Work on Your Branch Daily
# Start of every work session
git pull origin feature/branch_name   # Pull your own branch updates
#possible merge conflicts to deal with here if you haven't pulled in a while

# End of work session — save your progress
git add .
git commit -m "feat: added whisper medium model transcription"
git push origin feature/branch_name

# Step 3 — Keep Your Branch Up to Date with Main
git checkout main                            # Switch to main
git pull origin main                         # Get latest main
git checkout feature/branch_name     # Switch back to your branch
git fetch origin

---

# Step 4 — Merge Your Work into Main (Pull Request)
# Never merge directly into main from terminal. Always use a Pull Request on GitHub:
1. Push your branch → git push origin feature/whisper-integration
2. Go to GitHub → your repo
3. Click "Compare & Pull Request" (GitHub shows this automatically)
4. Add a title and description of what you did
5. Assign a teammate to review
6. Once approved → click "Merge Pull Request"
7. Delete the branch after merging to keep the repo clean

# Full Team Workflow Picture
          TEAMMATE 1                   YOU                      TEAMMATE 2
              │                         │                            │
    feature/summarizer        feature/whisper              feature/speaker
              │                         │                            │
         work & commit              work & commit              work & commit
              │                         │                            │
         push branch               push branch                push branch
              │                         │                            │
         Pull Request              Pull Request               Pull Request
              │                         │                            │
              └──────────── main ────────────────────────────────────┘
                         (always clean & stable)

# Step 5 — After Your Branch is Merged, Clean Up
git checkout main                               # Switch back to main
git pull origin main                            # Get the freshly merged code
git branch -d feature/whisper-integration       # Delete local branch











# ############## Best Practices for Avoiding Merge Conflicts

# Before You Start Working — Always Pull First
git stash                    # Save your local changes temporarily
git pull origin main         # Pull latest from teammate
git stash pop                # Reapply your changes on top

# When You're Ready to Push
git stash                    # Stash local changes
git pull origin main         # Pull any new teammate changes
git stash pop                # Reapply your changes
git add .                    # Stage everything
git commit -m "your message" # Commit
git push origin main         # Push to GitHub
```

### The Golden Rules

> 1. Always stash → pull → stash pop before pushing
> 2. Pull at the start of every work session — not just when pushing
> 3. Commit often — small commits are easier to merge than big ones

---

### Save This as a Cheatsheet
```
START OF DAY          PUSH YOUR WORK
──────────────        ──────────────
git stash             git stash
git pull origin main  git pull origin main
git stash pop         git stash pop
  ↓                   git add .
start coding          git commit -m "message"
                      git push origin main