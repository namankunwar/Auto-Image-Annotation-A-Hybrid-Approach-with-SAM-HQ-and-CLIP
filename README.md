SAM-HQ + CLIP Assisted Annotation Tool
A smart annotation tool that combines Meta's Segment Anything HQ (SAM-HQ) and OpenAI's CLIP to accelerate bounding box annotation with AI-assisted suggestions.

Key Features:
Hybrid Annotation Workflow: Start with a few manual boxes, let AI suggest the rest
CLIP-powered Similarity Search: Automatically finds visually similar objects
SAM-HQ Precision: Converts rough boxes into accurate object masks
Smart Filtering: Removes duplicates and maintains annotation quality
Interactive UI: Built with PyQt5 for seamless human-AI collaboration

Model Downloads:
You'll need to download these pretrained models:
SAM-HQ Tiny Model
Download: sam_hq_vit_tiny.pth
CLIP ViT-L/14 (Will auto-download on first run)
