// Ensure the DOM is fully loaded before executing
document.addEventListener('DOMContentLoaded', function() {
    const videoFeedImg = document.querySelector('.video-container img');
    const predictionOverlay = document.getElementById('prediction-overlay');
    const transcriptDiv = document.getElementById('transcript');

    // Check if the video feed image element exists
    if (videoFeedImg) {
        // The video feed is handled by the Flask route '/video_feed'
        // The <img> tag's src attribute is already set to this route in index.html
        // No additional JavaScript is needed here to start the video feed itself,
        // as the browser automatically requests the src.

        // However, we need to handle potential errors if the video feed fails to load.
        videoFeedImg.onerror = function() {
            console.error("Error loading video feed from /video_feed");
            // Display an error message or fallback in the UI
            if (predictionOverlay) {
                predictionOverlay.textContent = "Error loading webcam feed.";
                predictionOverlay.style.color = 'red';
            }
        };
    } else {
        console.error("Video feed image element not found in the DOM.");
        if (predictionOverlay) {
            predictionOverlay.textContent = "Webcam display area not found.";
            predictionOverlay.style.color = 'red';
        }
    }

    // --- Variables and Functions from index.html inline script ---

    const bufferSize = 8; // ~1.6 seconds
    const cooldownFrames = 8; // ~1.6 seconds cooldown after accepting a letter
    let ensembleBuffer = [];
    let lastAppendedLetter = ""; // This variable was in index.html but not used in the logic provided
    let currentWord = "";
    let transcriptText = ""; // This variable was in index.html but not used in the logic provided
    let cooldownCounter = 0;

    let rotateAngle = 0;
    let rotateAnimFrame = null;
    function startContinuousRotate() {
        const frame = document.querySelector('.camera-frame');
        if (!frame) return;
        function animate() {
            rotateAngle = (rotateAngle + 1.2) % 360; // ~60fps, 1.2deg per frame ≈ 2s per full rotation
            frame.style.setProperty('--rotate', rotateAngle + 'deg');
            rotateAnimFrame = requestAnimationFrame(animate);
        }
        if (!rotateAnimFrame) animate();
    }
    // Call this function when the DOM is ready
    startContinuousRotate();


    function triggerBorderAnimation() {
        const gradientContainer = document.querySelector('.gradient-container');
        if (!gradientContainer) return;
        gradientContainer.classList.add('green-border');
        setTimeout(() => {
            gradientContainer.classList.remove('green-border');
        }, 700);
    }

    let selectedMode = "ensemble"; // Keep this declaration
    const modeButtons = ["svm", "cnn", "ensemble"];
    function setMode(mode) {
        selectedMode = mode;
        modeButtons.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) {
                if (id === mode) {
                    btn.classList.add("active-mode");
                    btn.style.background = "#d0e6ff";
                    btn.style.color = "#3c67e3";
                } else {
                    btn.classList.remove("active-mode");
                    btn.style.background = "#e0e0e0";
                    btn.style.color = "#333";
                }
            }
        });
    }
    // Add event listeners for mode buttons
    modeButtons.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener("click", () => setMode(id));
        }
    });
    // Set initial mode
    setMode("ensemble");


    let lastTranscriptWord = ""; // Keep this declaration

    function createNewLineDiv(parentDiv) {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'transcript-line';
        parentDiv.appendChild(lineDiv);
        return lineDiv;
    }

    function updateTranscript() {
        const transcriptDiv = document.getElementById('transcript');
        if (!transcriptDiv) return;

        // Remove existing cursors
        const existingCursors = transcriptDiv.querySelectorAll('.cursor');
        existingCursors.forEach(cursor => cursor.remove());

        let currentLineDiv = transcriptDiv.lastElementChild;
         if (!currentLineDiv || currentLineDiv.className !== 'transcript-line') {
             currentLineDiv = createNewLineDiv(transcriptDiv); // Use the helper function
         }


        if (currentWord !== lastTranscriptWord) {
            // Clear the current line and set new content
            currentLineDiv.textContent = currentWord;

            // Add cursor after the text
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'cursor';
            currentLineDiv.appendChild(cursorSpan);

            lastTranscriptWord = currentWord;
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        } else if (!currentLineDiv.querySelector('.cursor')) {
            // Add cursor if it doesn't exist
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'cursor';
            currentLineDiv.appendChild(cursorSpan);
        }
    }


    function pollPredictions() {
        fetch(`/predictions?mode=${selectedMode}`)
            .then(response => {
                if (!response.ok) {
                    console.error(`HTTP error! status: ${response.status}`);
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Update overlay
                const overlay = document.getElementById("prediction-overlay");
                if (overlay && data.hand_detected && data.prediction) {
                     if (selectedMode === "ensemble") {
                        // Preserve the current color and text-shadow styles
                        const currentColor = overlay.style.color || '#41e33c';
                        const currentShadow = overlay.style.textShadow || '';

                        overlay.innerHTML = `
                            <span>${data.prediction}</span>
                            <span style="font-size:0.6em;display:block;margin-top:2px;">
                                SVM: ${(data.svm_confidence*100).toFixed(1)}%<br>
                                CNN: ${(data.cnn_confidence*100).toFixed(1)}%
                            </span>
                        `;

                        // Re-apply the current styles
                        overlay.style.color = currentColor;
                        overlay.style.textShadow = currentShadow;
                    } else {
                        // Preserve the current style but update the text
                        const currentText = data.prediction;
                        const currentColor = overlay.style.color || '#41e33c';
                        const currentShadow = overlay.style.textShadow || '';

                        overlay.textContent = currentText;
                        overlay.style.color = currentColor;
                        overlay.style.textShadow = currentShadow;
                    }
                } else if (overlay) {
                    overlay.textContent = "";
                }

                // Letter acceptance logic
                if (data.hand_detected && data.prediction) {
                    ensembleBuffer.push(data.prediction);
                    if (ensembleBuffer.length > bufferSize) {
                        ensembleBuffer.shift();
                    }
                    const allSame = ensembleBuffer.every(l => l === ensembleBuffer[0]);
                    const candidateLetter = allSame && ensembleBuffer.length === bufferSize ? ensembleBuffer[0] : "";

                    if (candidateLetter) {
                        if (cooldownCounter === 0) {
                            triggerBorderAnimation();
                            currentWord += candidateLetter;
                            cooldownCounter = cooldownFrames;
                            updateTranscript();
                        } else {
                            cooldownCounter--;
                        }
                    }
                } else {
                    // Clear the buffer and reset cooldown when hand is not detected
                    ensembleBuffer = [];
                    cooldownCounter = 0;
                }
            })
            .catch(err => {
                console.error("Error polling predictions:", err);
                ensembleBuffer = [];
                cooldownCounter = 0;
                const overlay = document.getElementById("prediction-overlay");
                if (overlay) overlay.textContent = "Prediction Error";
            });
    }

    // Start polling for predictions
    setInterval(pollPredictions, 200); // Poll every 200ms


    // Handle keyboard events for transcript
    document.addEventListener('keydown', (e) => {
        const transcriptDiv = document.getElementById('transcript');
        const currentLineDiv = transcriptDiv ? transcriptDiv.lastElementChild : null; // Check if transcriptDiv exists

        // Handle Space key - create new line
        if (e.code === 'Space') {
            // Check if current line is empty (excluding the cursor element)
            const isCurrentLineEmpty = !currentLineDiv ||
                (currentLineDiv.textContent.trim() === "" && !currentWord);

            // Don't create new line if current line is empty
            if (isCurrentLineEmpty) {
                e.preventDefault();
                return;
            }

            // Remove last word segment before adding new line
            if (currentLineDiv && lastTranscriptWord) {
                // Remove existing cursor
                const cursor = currentLineDiv.querySelector('.cursor');
                if (cursor) cursor.remove();

                currentLineDiv.textContent = currentLineDiv.textContent.slice(0, currentLineDiv.textContent.lastIndexOf(lastTranscriptWord));
                currentLineDiv.textContent += currentWord;
            }

            // Create new line div with cursor
            const newLineDiv = createNewLineDiv(transcriptDiv);
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'cursor';
            newLineDiv.appendChild(cursorSpan);

            currentWord = "";
            lastTranscriptWord = "";
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
            e.preventDefault();
        }

        // Handle Backspace key - delete last letter
        else if (e.code === 'Backspace') {
            // Only proceed if there's text to delete
            if (currentWord.length > 0) {
                // Remove the last character from currentWord
                currentWord = currentWord.slice(0, -1);

                // Update the display
                if (currentLineDiv) {
                    // Remove existing cursor
                    const cursor = currentLineDiv.querySelector('.cursor');
                    if (cursor) cursor.remove();

                    // Update text content
                    currentLineDiv.textContent = currentWord;

                    // Add cursor back
                    const cursorSpan = document.createElement('span');
                    cursorSpan.className = 'cursor';
                    currentLineDiv.appendChild(cursorSpan);

                    // Update lastTranscriptWord to match the new currentWord
                    lastTranscriptWord = currentWord;

                    // Trigger a visual feedback for deletion
                    triggerBorderAnimation();
                }
            }
            e.preventDefault();
        }
    });

    // Download transcript as .txt
    // This was inside a DOMContentLoaded listener in index.html, moving it here.
    document.getElementById('download-btn').addEventListener('click', function() {
        const transcriptDiv = document.getElementById('transcript');
        if (!transcriptDiv) return; // Add check
        const lines = Array.from(transcriptDiv.children).map(div => div.textContent);
        const text = lines.join('\n');
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'asl_transcript.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Filter mode toggle functionality
    // This was inside a DOMContentLoaded listener in index.html, moving it here.
    const filterModeToggle = document.getElementById('blend-mode-toggle');
    const videoImgElement = document.querySelector('.video-container img'); // Renamed to avoid conflict if videoImg is used elsewhere
    const overlayElement = document.getElementById('prediction-overlay'); // Renamed to avoid conflict

    // Function to update prediction overlay text style based on filter mode
    function updateOverlayStyle(isBlackAndWhite) {
        if (!overlayElement) return; // Add check
        if (isBlackAndWhite) {
            // White text with black border for B&W mode
            overlayElement.style.color = '#ffffff';
            overlayElement.style.textShadow = '2px 2px 2px #000, -2px -2px 2px #000, 2px -2px 2px #000, -2px 2px 2px #000';
        } else {
            // Green text with white border for negative mode
            overlayElement.style.color = '#41e33c';
            overlayElement.style.textShadow = ''; // Remove text shadow for negative mode
        }
    }

    // Initialize with negative mode (as checkbox is unchecked by default)
    if (videoImgElement) { // Add check
        videoImgElement.classList.add('negative-mode');
    }
    updateOverlayStyle(false);

    if (filterModeToggle && videoImgElement) { // Add checks
        filterModeToggle.addEventListener('change', function() {
            if (this.checked) {
                videoImgElement.classList.add('bw-mode');
                videoImgElement.classList.remove('negative-mode');
                updateOverlayStyle(true);
            } else {
                videoImgElement.classList.add('negative-mode');
                videoImgElement.classList.remove('bw-mode');
                updateOverlayStyle(false);
            }
        });
    }


    // Welcome modal functionality
    let currentModalPage = 1;
    const totalModalPages = 3;
    const welcomeModal = document.getElementById('welcome-modal'); // Get modal element

    function updatePaginationDots() {
        const dots = document.querySelectorAll('.pagination-dot');
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index + 1 === currentModalPage);
        });
    }

    function updateNavigationButtons() {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');

        if (!prevBtn || !nextBtn) return; // Add checks

        // Show/hide previous button
        prevBtn.style.visibility = currentModalPage > 1 ? 'visible' : 'hidden';

        // Change next button to checkmark on last page
        if (currentModalPage === totalModalPages) {
            nextBtn.innerHTML = '✓';
            nextBtn.onclick = closeModal;
        } else {
            nextBtn.innerHTML = '→';
            nextBtn.onclick = nextModalPage;
        }
    }

    function nextModalPage() {
        if (currentModalPage < totalModalPages) {
            const currentPageElement = document.getElementById(`modal-page-${currentModalPage}`);
            if (currentPageElement) currentPageElement.style.display = 'none'; // Add check
            currentModalPage++;
            const nextPageElement = document.getElementById(`modal-page-${currentModalPage}`);
            if (nextPageElement) nextPageElement.style.display = 'block'; // Add check
            updatePaginationDots();
            updateNavigationButtons();
        }
    }

    function prevModalPage() {
        if (currentModalPage > 1) {
            const currentPageElement = document.getElementById(`modal-page-${currentModalPage}`);
            if (currentPageElement) currentPageElement.style.display = 'none'; // Add check
            currentModalPage--;
            const prevPageElement = document.getElementById(`modal-page-${currentModalPage}`);
            if (prevPageElement) prevPageElement.style.display = 'block'; // Add check
            updatePaginationDots();
            updateNavigationButtons();
        }
    }

    function closeModal() {
        if (welcomeModal) welcomeModal.style.display = 'none'; // Add check
        // Optional: Set a flag in localStorage to avoid showing the modal on subsequent visits
        // localStorage.setItem('signSpeakIntroSeen', 'true');
    }

    // Initialize modal state on DOMContentLoaded
    updatePaginationDots();
    updateNavigationButtons();

    // ASL Chart Modal functionality
    const aslChartModal = document.getElementById('asl-chart-modal'); // Get modal element

    function showASLChart() {
        if (aslChartModal) aslChartModal.style.display = 'flex'; // Add check
    }

    function closeASLChartModal() {
        if (aslChartModal) aslChartModal.style.display = 'none'; // Add check
    }

    // Close modal when clicking outside of it
    window.onclick = function(event) {
        // Check if the clicked target is one of the modals
        if (welcomeModal && event.target === welcomeModal) {
             welcomeModal.style.display = 'none';
        }
        if (aslChartModal && event.target === aslChartModal) {
            aslChartModal.style.display = 'none';
        }
    }

    // Add event listener for the ASL chart button
    const aslChartBtn = document.getElementById('asl-chart-btn');
    if (aslChartBtn) { // Add check
        aslChartBtn.addEventListener('click', showASLChart);
    }

    // Add event listener for the ASL chart close button
    const aslChartCloseBtn = document.querySelector('.asl-chart-close');
     if (aslChartCloseBtn) { // Add check
        aslChartCloseBtn.addEventListener('click', closeASLChartModal);
    }


    // Check if intro has been seen before (optional feature)
    // window.addEventListener('DOMContentLoaded', function() { // This listener is redundant now
    //     if (localStorage.getItem('signSpeakIntroSeen')) {
    //         document.getElementById('welcome-modal').style.display = 'none';
    //     }
    // });

    // --- End of Variables and Functions from index.html inline script ---

    // The /preload request error suggests a fetch or similar call is made expecting JSON
    // but receiving HTML. This call is not explicitly in the inline script provided.
    // It might be a side effect of a library or another script.
    // Given the error message "Error preloading models", it's possible there was
    // intended client-side model loading that is now removed or misconfigured.
    // Since the server-side app.py loads the models, the client likely doesn't need
    // to preload them via a separate /preload endpoint.
    // I will not add any /preload fetch calls here. If the error persists after
    // consolidation, further investigation into other scripts or libraries will be needed.

}); // End of DOMContentLoaded listener

// The SyntaxError 'Identifier 'selectedMode' has already been declared'
// is resolved by having only one declaration within the DOMContentLoaded scope.

// The TypeError 'Cannot read properties of null (reading 'classList')' at index.html:1053
// is addressed by ensuring the script runs after DOMContentLoaded and adding checks
// before accessing properties of elements.