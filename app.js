window.onload = function() {
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
};

const canvas = document.getElementById('writeCanvas');
const context = canvas.getContext('2d');
let drawing = false;

canvas.addEventListener('mousedown', start);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stop);

document.getElementById('clearButton').addEventListener('click', clearCanvas);
document.getElementById('processButton').addEventListener('click', processDrawing);

function start(e) {
    drawing = true;
    context.beginPath();
    context.moveTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
}

function draw(e) {
    if (!drawing) {
        return;
    }
    
    context.lineWidth = 15;
    context.strokeStyle = 'white';
    context.lineCap = 'round';
    context.lineTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
    context.stroke();
}

function stop() {
    drawing = false;
}

function clearCanvas() {
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
}


function processDrawing() {
    let standardized = document.createElement('canvas');
    standardized.width = 28;
    standardized.height = 28;
    let resized = standardized.getContext('2d');
    resized.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

    const image = resized.getImageData(0, 0, 28, 28);
    const data = image.data; 

    let pixels = [];
    for (let y = 0; y < 28; y++) {
        let row = [];
        for (let x = 0; x < 28; x++) {
            let offset = (y * 28 + x) * 4;
            let red = data[offset];
            let green = data[offset + 1];
            let blue = data[offset + 2];
            let alpha = data[offset + 3];

            if (alpha > 0) {
                let rgb = 0.299 * red + 0.587 * green + 0.114 * blue;
                row.push(rgb < 128 ? 0 : 1); 
            } else {
                row.push(0); 
            }
        }
        pixels.push(row);
    }

    console.log(pixels);
    sendData(pixels);
}


function sendData(pixelData) {
    fetch('http://127.0.0.1:5000/predict', { //if the code is not working, you might need to change this url
    //run python app.py
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            pixels: pixelData
            //temporary data to test the code
            // pixels: [
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.7, 0.7, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7, 0.5, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.7, 0.7, 0, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7, 0.5, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.7, 0.7, 0, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7, 0.5, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 0.7, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7, 0.5, 0.2, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.5, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            // ]
        })
    })
    .then(response => response.json())
    .then(result => {
        //make sure to change the alert to text on the page
        alert('Prediction: ' + result.prediction);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

