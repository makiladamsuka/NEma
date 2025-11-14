import firebase from 'firebase/app';
import 'firebase/firestore';

const firebaseConfig = {
        apiKey: "AIzaSyC-IQ1YVjx9VgYiv6QAox6eBd-EZ4r7aaA",
        authDomain: "piwebrtc.firebaseapp.com",
        projectId: "piwebrtc",
        storageBucket: "piwebrtc.firebasestorage.app",
        messagingSenderId: "504393007442",
        appId: "1:504393007442:web:6e2749e901d581dbf37a8b",
        measurementId: "G-TGGBB2K38W"
        };

        
if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}
const firestore = firebase.firestore();

// --- WebRTC Setup ---
const servers = {
  iceServers: [
    {
      urls: ['stun:stun1.l.google.com:19302', 'stun:stun2.l.google.com:19302'],
    },
  ],
  iceCandidatePoolSize: 10,
};

const pc = new RTCPeerConnection(servers);
let localStream = null;
let remoteStream = null;

// --- HTML Element References ---
const webcamVideo = document.getElementById('webcamVideo');
const remoteVideo = document.getElementById('remoteVideo');
const webcamButton = document.getElementById('webcamButton');
const callButton = document.getElementById('callButton');
const answerButton = document.getElementById('answerButton');
const hangupButton = document.getElementById('hangupButton');
const callInput = document.getElementById('callInput');

// --- Helper Function for Button States ---
function setButtonsState(webcam, call, answer, hangup) {
    webcamButton.disabled = webcam;
    callButton.disabled = call;
    answerButton.disabled = answer;
    hangupButton.disabled = hangup;
}

// Initial state: Only webcam button enabled
setButtonsState(false, true, true, true); 

// --- Webcam Button Logic ---
webcamButton.onclick = async () => {
    // 1. Get the local camera and microphone stream
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });

    // 2. Push tracks from local stream to peer connection
    localStream.getTracks().forEach((track) => {
        pc.addTrack(track, localStream);
    });

    // 3. Show stream in HTML video
    webcamVideo.srcObject = localStream;

    // 4. Update button state
    setButtonsState(true, false, false, true);
};

// --- Remote Stream Setup ---
remoteStream = new MediaStream();
// Pull tracks from remote stream, add to video stream
pc.ontrack = event => {
    event.streams[0].getTracks().forEach(track => {
        remoteStream.addTrack(track);
    });
};
remoteVideo.srcObject = remoteStream;


// --- Call Button Logic (Caller) ---
callButton.onclick = async () => {
    // 1. Reference Firestore collections for signaling
    const callDoc = firestore.collection('calls').doc();
    const offerCandidates = callDoc.collection('offerCandidates');
    const answerCandidates = callDoc.collection('answerCandidates');

    callInput.value = callDoc.id;

    // 2. Get candidates for caller, save to db
    pc.onicecandidate = event => {
        event.candidate && offerCandidates.add(event.candidate.toJSON());
    };

    // 3. Create offer and set it as local description
    const offerDescription = await pc.createOffer();
    await pc.setLocalDescription(offerDescription);

    const offer = {
        sdp: offerDescription.sdp,
        type: offerDescription.type,
    };

    // 4. Save the offer to Firestore
    await callDoc.set({ offer });

    // 5. Listen for remote answer
    callDoc.onSnapshot((snapshot) => {
        const data = snapshot.data();
        if (!pc.currentRemoteDescription && data?.answer) {
            const answerDescription = new RTCSessionDescription(data.answer);
            pc.setRemoteDescription(answerDescription);
            setButtonsState(true, true, true, false); // Enable hangup on connection
        }
    });

    // 6. Listen for remote ICE candidates
    answerCandidates.onSnapshot(snapshot => {
        snapshot.docChanges().forEach((change) => {
            if (change.type === 'added') {
                const candidate = new RTCIceCandidate(change.doc.data());
                pc.addIceCandidate(candidate);
            }
        });
    });

    // Update button state (Call in progress)
    setButtonsState(true, true, true, false);
}


// --- Answer Button Logic (Receiver) ---
answerButton.onclick = async () => {
    const callId = callInput.value;
    const callDoc = firestore.collection('calls').doc(callId);
    const offerCandidates = callDoc.collection('offerCandidates');
    const answerCandidates = callDoc.collection('answerCandidates');

    // 1. Get candidates for answerer, save to db
    pc.onicecandidate = event => {
        event.candidate && answerCandidates.add(event.candidate.toJSON());
    };

    // 2. Fetch data, then set the offer
    const callData = (await callDoc.get()).data();
    if (!callData) {
        console.error("Call ID not found!");
        return;
    }

    const offerDescription = callData.offer;
    await pc.setRemoteDescription(new RTCSessionDescription(offerDescription));

    // 3. Create and set the answer
    const answerDescription = await pc.createAnswer();
    await pc.setLocalDescription(answerDescription);

    const answer = {
        type: answerDescription.type,
        sdp: answerDescription.sdp,
    };

    // 4. Update the Firestore document with the answer
    await callDoc.update({ answer });

    // 5. Listen to offer candidates
    offerCandidates.onSnapshot((snapshot) => {
        snapshot.docChanges().forEach((change) => {
            if (change.type === 'added') {
                let data = change.doc.data();
                pc.addIceCandidate(new RTCIceCandidate(data));
            }
        });
    });
    
    // Update button state (Call in progress)
    setButtonsState(true, true, true, false);
};


// --- Hangup Button Logic ---
hangupButton.onclick = () => {
    // 1. Stop all tracks and close connection
    localStream && localStream.getTracks().forEach(track => track.stop());
    pc.close();
    
    // 2. Clear video sources
    webcamVideo.srcObject = null;
    remoteVideo.srcObject = null;

    // 3. Clear call ID and reset buttons
    callInput.value = '';
    setButtonsState(false, true, true, true);
    
    // Note: To fully clean up, you should also delete the Firestore document associated with the call ID.
};