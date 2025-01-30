from multiprocessing import Process, Queue
import time

class PacketSynchronizer(Process):
    """
    A multiprocessing-based packet synchronizer that synchronizes
    context detector packets and gaze detector packets based on timestamps.
    """

    def __init__(self, context_queue, gaze_queue, output_queue, max_error=1.0, deletion_timeout=5.0):
        """
        Initializes the synchronizer as a multiprocessing process.

        Parameters:
            context_queue (Queue): Queue containing context packets [(time_stamp, final_label, transcription_text)].
            gaze_queue (Queue): Queue containing gaze packets [(gaze_time, gaze_score)].
            output_queue (Queue): Queue to store synchronized output 
                                  [(time_stamp, final_label, transcription_text, gaze_time, gaze_score)].
            max_error (float): Maximum allowed synchronization error in seconds (default: 1.0).
            deletion_timeout (float): Maximum time a packet can remain unmatched before removal (default: 5.0).
        """
        super().__init__()
        self.context_queue = context_queue
        self.gaze_queue = gaze_queue
        self.output_queue = output_queue
        self.max_error = max_error
        self.deletion_timeout = deletion_timeout
        self.running = True  # Control flag for stopping the process

    def stop(self):
        """Stops the synchronization process gracefully."""
        self.running = False

    def run(self):
        """
        Runs the synchronization loop, continuously matching context and gaze packets
        and pushing synchronized results to the output queue.
        """
        context_packets = []
        gaze_packets = []

        while self.running:
            # Retrieve new packets from queues
            new_context_packets = self._retrieve_packets(self.context_queue, context_packets)
            new_gaze_packets = self._retrieve_packets(self.gaze_queue, gaze_packets)

            # Determine the latest timestamp from both new packets
            latest_timestamp = self._get_latest_timestamp(new_context_packets, new_gaze_packets)

            # Remove old unmatched packets based on the latest timestamp
            if latest_timestamp is not None:
                context_packets = [(t, l, txt) for t, l, txt in context_packets if latest_timestamp - t <= self.deletion_timeout]
                gaze_packets = [(t, s) for t, s in gaze_packets if latest_timestamp - t <= self.deletion_timeout]

            # Synchronize packets if both lists have data
            if context_packets and gaze_packets:
                synchronized_data, context_packets, gaze_packets = self.synchronize(context_packets, gaze_packets)

                # Push synchronized results to the output queue
                for sync in synchronized_data:
                    try:
                        self.output_queue.put_nowait(sync)
                    except:
                        pass  # Avoid blocking if the queue is full

            time.sleep(0.01)  # Prevent CPU overload

    def _retrieve_packets(self, queue, packet_list):
        """Fetch new packets from the queue and add them to the list."""
        new_packets = []
        while not queue.empty():
            try:
                packet = queue.get_nowait()
                new_packets.append(packet)
                packet_list.append(packet)
            except:
                pass  # Skip if empty
        return new_packets

    def _get_latest_timestamp(self, new_context_packets, new_gaze_packets):
        """Returns the latest timestamp from new incoming packets."""
        latest_timestamps = [t for t, _, _ in new_context_packets] + [t for t, _ in new_gaze_packets]
        return max(latest_timestamps) if latest_timestamps else None

    def synchronize(self, context_packets, gaze_packets):
        """
        Synchronizes context packets and gaze packets based on timestamps.
        Removes synchronized packets and discards old unmatched ones.

        Parameters:
            context_packets (list of tuples): [(time_stamp, final_label, transcription_text), ...]
            gaze_packets (list of tuples): [(gaze_time, gaze_score), ...]

        Returns:
            (synchronized_data, updated_context_packets, updated_gaze_packets)
            synchronized_data: list of synchronized tuples [(time_stamp, final_label, transcription_text, gaze_time, gaze_score)]
            updated_context_packets: list of unsynchronized context packets
            updated_gaze_packets: list of unsynchronized gaze packets
        """
        synchronized_data = []
        new_context_packets = []
        new_gaze_packets = gaze_packets[:]  # Copy gaze_packets so we can modify the original list

        for time_stamp, final_label, transcription_text in context_packets:
            closest_gaze = None
            closest_diff = float("inf")
            best_gaze_index = None

            # Find the closest gaze packet within the max_error range
            for i, (gaze_time, gaze_score) in enumerate(new_gaze_packets):
                time_diff = abs(time_stamp - gaze_time)

                if time_diff <= self.max_error and time_diff < closest_diff:
                    closest_gaze = (gaze_time, gaze_score)
                    closest_diff = time_diff
                    best_gaze_index = i

                # Stop searching when moving past a possible match
                if gaze_time > time_stamp + self.max_error:
                    break

            # Append synchronized data and remove matched gaze packet
            if closest_gaze and best_gaze_index is not None:
                gaze_time, gaze_score = closest_gaze
                synchronized_data.append((time_stamp, final_label, transcription_text, gaze_time, gaze_score))
                del new_gaze_packets[best_gaze_index]  # Remove used gaze packet
            else:
                new_context_packets.append((time_stamp, final_label, transcription_text))  # Keep unmatched context packets

        return synchronized_data, new_context_packets, new_gaze_packets