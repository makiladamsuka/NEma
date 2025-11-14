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

webcamButton.onclick = async () => {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });

    // Push tracks from local stream to peer connection
    localStream.getTracks().forEach((track) => {
        pc.addTrack(track, localStream);
    });

    // Show stream in HTML video
    webcamVideo.srcObject = localStream;
}

remoteStream = new MediaStream();

// Pull tracks from remote stream, add to video stream
pc.ontrack = event => {
    event.streams[0].getTracks().forEach(track => {
        remoteStream.addTrack(track);
    });
};

remoteVideo.srcObject = remoteStream;

callButton.onclick = async () => {
// Reference Firestore collections for signaling
  const callDoc = firestore.collection('calls').doc();
  const offerCandidates = callDoc.collection('offerCandidates');
  const answerCandidates = callDoc.collection('answerCandidates');

  callInput.value = callDoc.id;

  // Get candidates for caller, save to db
  pc.onicecandidate = event => {
    event.candidate && offerCandidates.add(event.candidate.toJSON());
  };

  // Create offer
  const offerDescription = await pc.createOffer();
  await pc.setLocalDescription(offerDescription);

  const offer = {
    sdp: offerDescription.sdp,
    type: offerDescription.type,
  };

  await callDoc.set({ offer });

  // Listen for remote answer
  callDoc.onSnapshot((snapshot) => {
    const data = snapshot.data();
    if (!pc.currentRemoteDescription && data?.answer) {
      const answerDescription = new RTCSessionDescription(data.answer);
      pc.setRemoteDescription(answerDescription);
    }
  });

  // Listen for remote ICE candidates
  answerCandidates.onSnapshot(snapshot => {
    snapshot.docChanges().forEach((change) => {
      if (change.type === 'added') {
        const candidate = new RTCIceCandidate(change.doc.data());
        pc.addIceCandidate(candidate);
      }
    });
  });
}

answerButton.onclick = async () => {
  const callId = callInput.value;
  const callDoc = firestore.collection('calls').doc(callId);
  const offerCandidates = callDoc.collection('offerCandidates');
  const answerCandidates = callDoc.collection('answerCandidates');

  pc.onicecandidate = event => {
    event.candidate && answerCandidates.add(event.candidate.toJSON());
  };

  // Fetch data, then set the offer & answer

  const callData = (await callDoc.get()).data();

  const offerDescription = callData.offer;
  await pc.setRemoteDescription(new RTCSessionDescription(offerDescription));

  const answerDescription = await pc.createAnswer();
  await pc.setLocalDescription(answerDescription);

  const answer = {
    type: answerDescription.type,
    sdp: answerDescription.sdp,
  };

  await callDoc.update({ answer });

  // Listen to offer candidates

  offerCandidates.onSnapshot((snapshot) => {
    snapshot.docChanges().forEach((change) => {
      console.log(change)
      if (change.type === 'added') {
        let data = change.doc.data();
        pc.addIceCandidate(new RTCIceCandidate(data));
      }
    });
  });
};



// Get all the HTML elements
const webcamVideo = document.getElementById('webcamVideo');
const remoteVideo = document.getElementById('remoteVideo');
const webcamButton = document.getElementById('webcamButton');
const callButton = document.getElementById('callButton');
const answerButton = document.getElementById('answerButton');
const hangupButton = document.getElementById('hangupButton');
const callInput = document.getElementById('callInput');

// Function to update the disabled state of buttons
function setButtonsState(webcam, call, answer, hangup) {
    webcamButton.disabled = webcam;
    callButton.disabled = call;
    answerButton.disabled = answer;
    hangupButton.disabled = hangup;
}

// Initial state: Only webcam button enabled
setButtonsState(false, true, true, true);

// When webcam starts, enable Call/Answer buttons
webcamButton.onclick = async () => {
    // ... (Your existing webcamButton code here) ...
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localStream.getTracks().forEach((track) => {
        pc.addTrack(track, localStream);
    });
    webcamVideo.srcObject = localStream;
    
    // Update button state
    setButtonsState(true, false, false, true); 
};

// When call is created, enable Hangup
callButton.onclick = async () => {
    // ... (Your existing callButton code here) ...
    // Update button state
    setButtonsState(true, true, true, false); 
};

// When call is answered, enable Hangup
answerButton.onclick = async () => {
    // ... (Your existing answerButton code here) ...
    // Update button state
    setButtonsState(true, true, true, false);
};


// Add hangup logic
hangupButton.onclick = () => {
    pc.close(); // Close the peer connection
    localStream && localStream.getTracks().forEach(track => track.stop()); // Stop webcam
    webcamVideo.srcObject = null;
    remoteVideo.srcObject = null;
    
    // Reset buttons and connection
    setButtonsState(false, true, true, true);
    // You may also want to delete the Firestore document for cleanup
};
