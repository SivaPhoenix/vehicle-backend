const express = require('express');
const cors = require('cors'); // Import the cors package
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const app = express();
const port = 3001;

// Use CORS middleware
app.use(cors());

// Middleware to parse JSON and form data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('video'), (req, res) => {
    const videoPath = req.file.path;
    const outputVideoPath = path.join('uploads', 'output.mp4');

    // Call the Python script to process the video
    exec(`python process_video.py "${videoPath}" "${outputVideoPath}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send('Error processing video');
        }
        console.log(`stdout: ${stdout}`);
        console.error(`stderr: ${stderr}`);
        
        res.json({ outputVideoPath });
    });
});

app.use('/uploads', express.static('uploads'));

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
